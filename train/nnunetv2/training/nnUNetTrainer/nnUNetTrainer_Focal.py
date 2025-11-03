# Filename: nnUNetTrainer_Focal.py
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np # Required for weight calculations

# --- Relative Import Strategy ---
try:
    from .nnUNetTrainer import nnUNetTrainer
except ImportError:
    try:
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("ERROR: Cannot import base nnUNetTrainer."); raise
# --------------------------------

# Import necessary standard components from nnU-Net v2
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss


# ====================================================================================
# Self-Contained Focal Loss Implementation
# ====================================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss implementation robustly handling multi-class (Softmax) and multi-label (Sigmoid) scenarios.
    """
    def __init__(self, gamma=2.0, apply_nonlin=None, ignore_index=None):
        super().__init__()
        self.gamma = gamma
        self.apply_nonlin = apply_nonlin
        # This attribute is set by the trainer after initialization
        self.apply_nonlin_type = None 
        self.ignore_index = ignore_index

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        
        if self.apply_nonlin is None or self.apply_nonlin_type is None:
             raise ValueError("FocalLoss requires apply_nonlin and apply_nonlin_type to be specified.")

        # 1. Apply nonlinearity to get probabilities
        probs = self.apply_nonlin(input)

        # 2. Handle target shape and type
        if self.apply_nonlin_type == 'softmax':
            # Multi-class (CE): Requires long indices
            if target.ndim == input.ndim and target.shape[1] == 1:
                target = target.squeeze(1)
            target = target.long()
        else:
            # Multi-label (BCE): Requires float targets
            target = target.float()

        # 3. Calculate the standard loss (CE or BCE) and probability of true class (pt) element-wise

        if self.apply_nonlin_type == 'sigmoid':
            # Multi-label (BCE)
             ce_loss = F.binary_cross_entropy(probs, target, reduction='none')
             pt = torch.where(target == 1, probs, 1 - probs)

        else:
            # Multi-class (CE/NLL)
            log_probs = torch.log(probs.clip(min=1e-7))
            
            # Calculate NLL Loss (element-wise), handling ignore_index
            ignore_idx_val = self.ignore_index if self.ignore_index is not None else -100
            ce_loss = F.nll_loss(log_probs, target, reduction='none', ignore_index=ignore_idx_val)

            # Calculate pt for CE: Gather the probability of the true class
            target_expanded = target.unsqueeze(1)
            
            # Robust gathering handling potential ignore indices
            valid_mask = torch.ones_like(target, dtype=torch.bool)
            if self.ignore_index is not None:
                valid_mask = (target != self.ignore_index)
            
            # Create a safe target tensor for gathering
            safe_target_expanded = target_expanded.clone()
            
            # Ensure the safe index is within bounds [0, C-1].
            if probs.shape[1] > 0:
                 # Clamp indices to the valid range [0, C-1]
                 safe_target_expanded = torch.clamp(safe_target_expanded, 0, probs.shape[1] - 1)
            
            # Replace invalid indices with a safe index (e.g., 0) before gathering
            safe_target_expanded[~valid_mask.unsqueeze(1)] = 0
            
            pt = torch.gather(probs, 1, safe_target_expanded).squeeze(1)
            
            # Handle ignore index for pt: If ignored, pt should be 1 so the focal term (1-pt) is 0.
            pt = pt.masked_fill(~valid_mask, 1.0)


        # 4. Calculate the Focal Term: (1 - pt)^gamma
        focal_term = (1.0 - pt) ** self.gamma

        # 5. Combine: Focal Loss = Focal Term * CE Loss
        loss = focal_term * ce_loss
        
        # 6. Final reduction (mean)
        return loss.mean()

# ====================================================================================
# Combined Loss Wrapper
# ====================================================================================

class DC_and_Focal_loss(nn.Module):
    """
    Wrapper class to combine Dice Loss and Focal Loss.
    """
    def __init__(self, focal_loss_kwargs, dice_loss_kwargs, weight_focal=1.0, weight_dice=1.0):
        super(DC_and_Focal_loss, self).__init__()
        
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

        # Initialize the individual loss components
        self.focal_loss = FocalLoss(**focal_loss_kwargs)
        self.dc_loss = MemoryEfficientSoftDiceLoss(**dice_loss_kwargs)

    def forward(self, net_output, target):
        # Calculate individual losses
        dc_loss = self.dc_loss(net_output, target)
        focal_loss = self.focal_loss(net_output, target)
        
        # Combine losses
        result = self.weight_focal * focal_loss + self.weight_dice * dc_loss
        return result

# ====================================================================================
# Trainer Implementation
# ====================================================================================

class nnUNetTrainer_Focal(nnUNetTrainer):
    
    # Explicit __init__ signature to avoid introspection errors
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Initialized nnUNetTrainer_Focal (Dice + Self-Contained Focal Loss).")

    # Override the internal loss builder method
    def _build_loss(self):
        # Configuration for Focal Loss
        gamma = 2.0
        
        # Determine the appropriate nonlinearity
        if self.label_manager.has_regions:
             # Multi-label segmentation
             nonlin = torch.sigmoid
             nonlin_type = 'sigmoid'
        else:
             # Multi-class segmentation
             nonlin = lambda x: torch.softmax(x, dim=1)
             nonlin_type = 'softmax'

        # --- Configure Dice Loss ---
        # Replicate Dice Configuration Logic manually
        
        if self.label_manager.has_regions:
            ignore_background_in_dice = False
        else:
            # Standard multi-class behavior: ignore background (0) if it's not the ignore label.
            if self.label_manager.ignore_label != 0 and 0 in self.label_manager.all_labels:
                 ignore_background_in_dice = True
            else:
                 ignore_background_in_dice = False

        # Assemble the configuration dictionary for Dice Loss
        dice_kwargs = {
            'apply_nonlin': nonlin,
            'batch_dice': self.configuration_manager.batch_dice,
            'smooth': 1e-5,
            'do_bg': not ignore_background_in_dice,
            'ddp': self.is_ddp,
        }
        
        # --- Configure Focal Loss ---
        focal_kwargs = {
            'apply_nonlin': nonlin, 
            'gamma': gamma,
            'ignore_index': self.label_manager.ignore_label if self.label_manager.ignore_label is not None else None
        }

        self.print_to_log_file(f"Using Focal Loss. Gamma={gamma}. Nonlinearity: {nonlin_type}.")

        # --- Instantiate Combined Loss Wrapper ---
        loss = DC_and_Focal_loss(focal_kwargs, dice_kwargs, weight_focal=1.0, weight_dice=1.0)
        
        # Set the nonlin_type attribute required by the FocalLoss component
        loss.focal_loss.apply_nonlin_type = nonlin_type

        # Wrap for deep supervision if enabled
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            
            # FIX: Replicate Deep Supervision Weighting Logic (Resolves AttributeError)
            # Since self._get_deep_supervision_weights is unavailable, we replicate the standard weighting scheme.

            # Standard nnU-Net weighting: Weights decrease by a factor of 2 for each subsequent scale.
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

            # Handle potential masks defined in the plans (e.g., for cascaded models)
            # We need to inspect the plans_manager attributes robustly.
            
            mask = None
            # Check if plans attribute exists and contains the necessary key
            if hasattr(self.plans_manager, 'plans') and isinstance(self.plans_manager.plans, dict):
                 mask = self.plans_manager.plans.get('deep_supervision_masks')
            
            if mask is not None:
                # Ensure mask length matches weights length
                if len(mask) != len(weights):
                     self.print_to_log_file(f"WARNING: Length mismatch in deep_supervision_masks. Expected {len(weights)}, got {len(mask)}. Adjusting.")
                     # Truncate mask if too long, or pad if too short
                     if len(mask) > len(weights):
                        mask = mask[:len(weights)]
                     else:
                        # Padding with 1s if mask is shorter (unusual configuration)
                        pad_len = len(weights) - len(mask)
                        mask = np.pad(mask, (0, pad_len), 'constant', constant_values=1)

                weights = weights * mask

            # Normalize weights so they sum to 1
            if weights.sum() > 0:
                weights = weights / weights.sum()
            else:
                # Safety check if all weights are zero
                weights = np.ones_like(weights) / len(weights)

            # ------------------------------------------------------------------

            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss
