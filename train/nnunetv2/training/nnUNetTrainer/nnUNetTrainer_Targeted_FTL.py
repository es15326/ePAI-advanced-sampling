# Filename: nnUNetTrainer_Targeted_FTL.py
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os

# (Standard Imports...)
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.utilities.ddp_allgather import AllGatherGrad 
# Import LambdaLR for custom scheduler implementation
from torch.optim.lr_scheduler import LambdaLR

try:
    from nnunetv2.utilities.helpers import softmax_helper_dim1
except ImportError:
    softmax_helper_dim1 = lambda x: torch.softmax(x, dim=1)

try:
    from .nnUNetTrainer_ModeratedAug_Targeted import nnUNetTrainer_ModeratedAug_Targeted as BaseTrainer
except ImportError:
    print("ERROR: Cannot import BaseTrainer (nnUNetTrainer_ModeratedAug_Targeted)."); raise

# =================================================================================
# HELPER FUNCTION FOR LR SCHEDULER (Standard nnU-Net Polynomial Decay)
# =================================================================================
def poly_lr_factor(epoch, max_epochs, exponent=0.9):
    # Calculates the LR decay factor (multiplicative factor for LambdaLR).
    # Replicates standard nnU-Net behavior.
    if epoch > max_epochs:
        return 0.0
    return (1 - epoch / max_epochs)**exponent

# =================================================================================
# FOCAL TVERSKY LOSS IMPLEMENTATION
# =================================================================================
class FocalTverskyLoss(nn.Module):
    # (Implementation identical to the previous functional version)
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.333, smooth=1e-5, batch_dice=False, do_bg=False, apply_nonlin=None, ignore_label=None):
        super(FocalTverskyLoss, self).__init__()
        self.gamma = gamma; self.smooth = smooth; self.batch_dice = batch_dice; self.do_bg = do_bg
        self.apply_nonlin = apply_nonlin; self.ignore_label = ignore_label

        if not np.isclose(alpha + beta, 1.0):
             print(f"INFO: FocalTverskyLoss: alpha ({alpha:.4f}) + beta ({beta:.4f}) normalized.")
             total = alpha + beta; self.alpha = alpha / total; self.beta = beta / total
        else:
            self.alpha = alpha; self.beta = beta

    def forward(self, net_output: torch.Tensor, target: torch.Tensor):
        if self.apply_nonlin is not None: net_output = self.apply_nonlin(net_output)
        shp_x = net_output.shape
        if target.ndim == len(shp_x) - 1: target = target.unsqueeze(1)
        
        mask = None
        if self.ignore_label is not None:
            mask = (target != self.ignore_label).float(); target = target.clone()
            target[target == self.ignore_label] = 0 

        y_onehot = torch.zeros(shp_x, device=net_output.device, dtype=torch.float32)
        y_onehot.scatter_(1, target.long(), 1)
        
        if mask is not None:
            if mask.shape[1] == 1: mask = mask.expand_as(net_output)
            net_output = net_output * mask; y_onehot = y_onehot * mask

        axes = tuple(range(2, len(shp_x))); p = net_output; g = y_onehot
        tp = (p * g).sum(axes); fn = ((1 - p) * g).sum(axes); fp = (p * (1 - g)).sum(axes)

        if self.batch_dice:
            if torch.distributed.is_initialized():
                tp = AllGatherGrad.apply(tp).sum(0); fn = AllGatherGrad.apply(fn).sum(0); fp = AllGatherGrad.apply(fp).sum(0)
            else:
                tp = tp.sum(0); fn = fn.sum(0); fp = fp.sum(0)

        tversky_index = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        loss = torch.pow(1.0 - tversky_index, self.gamma)

        if not self.do_bg:
            if self.batch_dice: loss = loss[1:]
            else: loss = loss[:, 1:]
        return loss.mean()

# =================================================================================
# TRAINER IMPLEMENTATION (Corrected configure_optimizers)
# =================================================================================

class nnUNetTrainer_Targeted_FTL(BaseTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        
        # (DDP/batch_dice fix remains the same)
        if not torch.distributed.is_initialized():
            if configuration in plans.get('configurations', {}):
                if plans['configurations'][configuration].get('batch_dice', False):
                    print(f"INFO: DDP not initialized. Modifying plans to set batch_dice=False.")
                    plans['configurations'][configuration]['batch_dice'] = False

        # Initialize the parent (BaseTrainer)
        super().__init__(plans, configuration, fold, dataset_json, device)

        # Set defaults
        self.ftl_alpha = 0.7; self.ftl_beta = 0.3; self.ftl_gamma = 1.333
        self.initial_lr = 0.01; self.weight_decay = 3e-5; self.optimizer_type = "SGD"
        
        # Extended Training
        self.num_epochs = 1500

        # --- HYPERPARAMETER TUNING (FTL and Optimizer) ---
        try:
            # FTL Parameters
            self.ftl_alpha = float(os.environ.get("FTL_ALPHA", 0.7))
            self.ftl_beta = float(os.environ.get("FTL_BETA", 0.3))
            self.ftl_gamma = float(os.environ.get("FTL_GAMMA", 1.333))
            
            # Optimizer Parameters
            self.initial_lr = float(os.environ.get("INIT_LR", 0.01))
            self.weight_decay = float(os.environ.get("WEIGHT_DECAY", 3e-5))
            self.optimizer_type = os.environ.get("OPTIMIZER_TYPE", "SGD")
            
            # Validation
            if self.optimizer_type not in ["SGD", "AdamW"]:
                self.print_to_log_file(f"WARNING: Invalid OPTIMIZER_TYPE '{self.optimizer_type}'. Defaulting to SGD.")
                self.optimizer_type = "SGD"

            # Log overrides
            if any(k in os.environ for k in ["FTL_ALPHA", "FTL_BETA", "FTL_GAMMA", "INIT_LR", "WEIGHT_DECAY", "OPTIMIZER_TYPE"]):
                self.print_to_log_file("INFO: FTL/Optimizer Hyperparameters overridden by environment variables.")

        except ValueError as e:
            self.print_to_log_file(f"ERROR parsing HParam environment variables. Using defaults. Error: {e}")

        # -------------------------------------------------------
        
        self.print_to_log_file("Initialized nnUNetTrainer_Targeted_FTL.")
        self.print_to_log_file(f"  Loss Config: Alpha={self.ftl_alpha:.4f}, Beta={self.ftl_beta:.4f}, Gamma={self.ftl_gamma:.4f}.")
        self.print_to_log_file(f"  Optimizer Config: Type={self.optimizer_type}, Initial LR={self.initial_lr:.5f}, Weight Decay={self.weight_decay:.6f}.")
        self.print_to_log_file(f"Extended Training Active: num_epochs set to {self.num_epochs}.")


    # Override the configure_optimizers method (CORRECTED)
    def configure_optimizers(self):
        if self.optimizer_type == "SGD":
            optimizer = torch.optim.SGD(
                self.network.parameters(), 
                self.initial_lr,
                weight_decay=self.weight_decay,
                momentum=0.99, 
                nesterov=True
            )
        elif self.optimizer_type == "AdamW":
            optimizer = torch.optim.AdamW(
                self.network.parameters(), 
                self.initial_lr,
                weight_decay=self.weight_decay
            )
        else:
            raise RuntimeError(f"Unsupported optimizer type: {self.optimizer_type}")

        # --- FIX: Implement LR Scheduler directly using LambdaLR ---
        # This resolves the AttributeError by implementing the logic instead of calling _get_lr_scheduler().
        
        # Define the lambda function for polynomial decay using the helper function.
        # LambdaLR expects a function that returns the multiplicative factor.
        lr_lambda = lambda epoch: poly_lr_factor(epoch, self.num_epochs, exponent=0.9)
        
        # Create the LambdaLR scheduler
        lr_scheduler = LambdaLR(optimizer, lr_lambda)
        # ---------------------------------------------------------------------------------

        return optimizer, lr_scheduler

    def _build_loss(self):
        # (Loss building logic remains the same)
        if self.label_manager.has_regions:
            nonlin = torch.sigmoid; do_bg = True 
        else:
            nonlin = softmax_helper_dim1; do_bg = False 

        loss = FocalTverskyLoss(
            alpha=self.ftl_alpha, beta=self.ftl_beta, gamma=self.ftl_gamma,
            batch_dice=self.configuration_manager.batch_dice, do_bg=do_bg,
            apply_nonlin=nonlin, smooth=1e-5, ignore_label=self.label_manager.ignore_label
        )
        
        self.print_to_log_file(f"Loss function override initialized. Batch_dice={self.configuration_manager.batch_dice}.")

        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = torch.tensor(np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))]),
                                   dtype=torch.float32, device=self.device)
            weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
            
        return loss
