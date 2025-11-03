# Filename: nnUNetTrainer_Targeted_Focal.py
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

# Import essential components
from nnunetv2.training.loss.dice import SoftDiceLoss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper

# Import efficient softmax helper if available, otherwise define fallback
try:
    from nnunetv2.utilities.helpers import softmax_helper_dim1
except ImportError:
    # Fallback if helper is not available in this version
    print("INFO: softmax_helper_dim1 not found, using torch.softmax fallback.")
    softmax_helper_dim1 = lambda x: torch.softmax(x, dim=1)

# Import the base trainer (Moderated Augmentation + Targeted Sampling)
try:
    # Ensure nnUNetTrainer_ModeratedAug_Targeted.py is present in the same directory
    from .nnUNetTrainer_ModeratedAug_Targeted import nnUNetTrainer_ModeratedAug_Targeted as BaseTrainer
except ImportError:
    print("ERROR: Cannot import BaseTrainer (nnUNetTrainer_ModeratedAug_Targeted). Ensure the file is present."); raise


# =================================================================================
# EMBEDDED FOCAL LOSS IMPLEMENTATION
# =================================================================================

class FocalLoss(nn.Module):
    """
    Efficient implementation of Focal Loss based on Cross-Entropy.
    It operates directly on logits (network output before activation).
    Note: This implementation is designed for standard multi-class segmentation (Softmax/CE).
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', ignore_index=None, **kwargs):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha # Class weights (optional, typically None)
        self.reduction = reduction
        self.ignore_index = ignore_index
        # Ignore apply_nonlin if passed in kwargs, as this implementation uses logits.
        if 'apply_nonlin' in kwargs and kwargs['apply_nonlin'] is not None:
            print("INFO: apply_nonlin is ignored in embedded FocalLoss implementation as it operates on logits.")

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Input (logits): [B, C, X, Y, Z]
        # Target (class indices): [B, X, Y, Z] (Long type)
        
        # 1. Calculate the standard Cross-Entropy Loss per element.
        # F.cross_entropy efficiently handles logits and the ignore_index.
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index if self.ignore_index is not None else -100)

        # 2. Calculate probabilities (pt)
        # We use the mathematical identity: pt = exp(-ce_loss). This is efficient and numerically stable.
        pt = torch.exp(-ce_loss)

        # 3. Calculate the Focal modulating factor (1 - pt)^gamma
        focal_factor = (1 - pt) ** self.gamma

        # 4. Apply the modulating factor to the loss
        focal_loss = focal_factor * ce_loss

        # 5. Handle class weights (Alpha) if provided (Omitted as alpha=None is used)

        # 6. Apply reduction (mean/sum)
        # We must ensure we only average over valid (non-ignored) elements.
        if self.reduction == 'mean':
            if self.ignore_index is not None:
                mask = target != self.ignore_index
                if mask.sum() > 0:
                    return focal_loss[mask].mean()
                else:
                    # Handle case where the entire batch/image is ignored
                    return torch.tensor(0.0, device=input.device, requires_grad=True)
            return focal_loss.mean()
        elif self.reduction == 'sum':
            if self.ignore_index is not None:
                mask = target != self.ignore_index
                return focal_loss[mask].sum()
            return focal_loss.sum()
        else:
            return focal_loss

# =================================================================================
# END EMBEDDED FOCAL LOSS
# =================================================================================


# Define the manual combination wrapper
class DC_and_FL_Loss_Manual(nn.Module):
    """
    Manually combines SoftDiceLoss and the embedded FocalLoss.
    """
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_dice=1, weight_focal=1):
        super(DC_and_FL_Loss_Manual, self).__init__()
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        # Initialize losses.
        self.dc = SoftDiceLoss(**soft_dice_kwargs)
        # Initialize the embedded FocalLoss
        self.fl = FocalLoss(**focal_kwargs)

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        # 1. Calculate Dice loss
        # Dice loss handles the [B, 1, X, Y, Z] target format correctly.
        dc_loss = self.dc(net_output, target)

        # 2. Prepare target for Focal Loss.
        # FocalLoss expects [B, X, Y, Z] (Long type).
        # The input target from the dataloader is usually [B, 1, X, Y, Z] (Float type).
        target_fl = target
        
        # Check if the target has the same number of dimensions as output and a singleton channel dimension
        # This handles inputs from different deep supervision levels correctly.
        if target_fl.ndim == net_output.ndim and target_fl.shape[1] == 1:
           # Squeeze the channel dimension
           target_fl = target_fl[:, 0]
        
        # Ensure the type is Long for class indices
        target_fl = target_fl.long()

        # 3. Calculate Focal loss
        fl_loss = self.fl(net_output, target_fl)

        # 4. Combine losses
        result = self.weight_dice * dc_loss + self.weight_focal * fl_loss
        return result


class nnUNetTrainer_Targeted_Focal(BaseTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        
        # --- FIX FOR DDP/batch_dice ERROR (AttributeError) ---
        # If running without DDP (torch.distributed.is_initialized() is False), batch_dice=True causes an error 
        # because it attempts distributed communication (AllGather) which is not initialized.
        # We must modify the plans BEFORE super().__init__ because ConfigurationManager properties become read-only after initialization.
        if not torch.distributed.is_initialized():
            # Check if the configuration exists in the plans dictionary
            if configuration in plans.get('configurations', {}):
                # Check if batch_dice is currently True
                if plans['configurations'][configuration].get('batch_dice', False):
                    # Use print() here as the logger (self.print_to_log_file) is not yet initialized.
                    print(f"INFO: DDP not initialized. Modifying plans to set batch_dice=False for configuration '{configuration}' to prevent communication errors.")
                    # Modify the dictionary in place
                    plans['configurations'][configuration]['batch_dice'] = False
            else:
                    # Handle unexpected case where configuration is missing
                    print(f"WARNING: Configuration '{configuration}' not found in plans. Cannot check/set batch_dice.")
        # ------------------------------------------------------

        # Initialize the parent (BaseTrainer) with the potentially modified plans
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.print_to_log_file("Initialized nnUNetTrainer_Targeted_Focal (Self-Contained Focal Loss).")
        self.print_to_log_file("Loss Strategy: Dice + Embedded Focal Loss.")

    def _build_loss(self):
        """
        Manually combine Dice and the embedded Focal Loss.
        """
        focal_gamma = 2.0
        
        # --- Compatibility Check and Configuration ---
        # The embedded FocalLoss relies on Cross-Entropy (Softmax). 
        if self.label_manager.has_regions:
             # It is not compatible with BCE (Sigmoid) used for regions (multi-label).
             raise RuntimeError("The embedded FocalLoss implementation in this trainer is designed for multi-class (Softmax) and is not compatible with region-based segmentation (has_regions=True/Sigmoid).")

        # Since has_regions is False, we use Softmax and exclude background for Dice.
        nonlin = softmax_helper_dim1
        do_bg_dice = False
        # -----------------------------------------------------------

        # Configure Dice Loss kwargs
        # Dice needs the activation function to convert logits to probabilities.
        # We use the batch_dice setting from the configuration_manager (which now reflects the modified plans).
        soft_dice_kwargs = {
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': do_bg_dice,
            'apply_nonlin': nonlin, 
        }

        # Configure Focal Loss kwargs
        # Focal Loss operates on logits internally, so apply_nonlin is not needed/used.
        focal_kwargs = {
            'gamma': focal_gamma,
            'alpha': None
        }
        
        # Handle ignore label if present
        if self.label_manager.ignore_label is not None:
            soft_dice_kwargs['ignore_label'] = self.label_manager.ignore_label
            # The embedded FocalLoss uses 'ignore_index'
            focal_kwargs['ignore_index'] = self.label_manager.ignore_label

        # Initialize the manual wrapper
        loss = DC_and_FL_Loss_Manual(
            soft_dice_kwargs=soft_dice_kwargs,
            focal_kwargs=focal_kwargs,
            weight_dice=1,
            weight_focal=1
        )

        self.print_to_log_file(f"Loss function override: Manual combination of SoftDiceLoss (Softmax) and Embedded FocalLoss (Gamma={focal_gamma}). Batch_dice={self.configuration_manager.batch_dice}.")

        # Handle Deep Supervision (Standard logic)
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = torch.tensor(np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))]),
                                   dtype=torch.float32, device=self.device)
            weights = weights / weights.sum()
            
            # Wrap the combined loss for deep supervision
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss
