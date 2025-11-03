# Filename: nnUNetTrainer_Targeted_FTL_SWA.py
import torch
import os # Import os module
from torch.optim.swa_utils import AveragedModel, SWALR
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import the updated FTL trainer
try:
    from .nnUNetTrainer_Targeted_FTL import nnUNetTrainer_Targeted_FTL as BaseTrainer
except ImportError:
    print("ERROR: Cannot import BaseTrainer (nnUNetTrainer_Targeted_FTL)."); raise

class nnUNetTrainer_Targeted_FTL_SWA(BaseTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        # BaseTrainer (FTL) handles FTL, Optimizer, Sampling, Augmentation HParams
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- SWA Configuration & Hyperparameter Tuning (SWA_LR only) ---
        self.use_swa = True
        self.swa_start_epoch = int(self.num_epochs * 0.80)
        
        # Set default
        self.swa_lr = 1e-3

        # Read SWA LR from environment variable
        try:
            self.swa_lr = float(os.environ.get("SWA_LR", 1e-3))
            if "SWA_LR" in os.environ:
                 self.print_to_log_file(f"HParam Override: SWA_LR set to {self.swa_lr}")
        except ValueError as e:
            self.print_to_log_file(f"ERROR parsing SWA_LR environment variable. Using default (1e-3). Error: {e}")
            
        # -------------------------------------------------

        self.swa_model = None
        self.swa_scheduler = None
        self.print_to_log_file("Initialized nnUNetTrainer_Targeted_FTL_SWA.")
        # Log the complete configuration (FTL/Optimizer parameters are logged by the parent)
        self.print_to_log_file(f"  SWA Config: Start Epoch={self.swa_start_epoch}, SWA LR={self.swa_lr:.5f}.")

    # NOTE: We do NOT override configure_optimizers here, as the parent (FTLTrainer) now handles it.
    
    # (The rest of the SWA implementation remains the same)
    
    def initialize(self):
        super().initialize()
        if self.use_swa:
            self.initialize_swa()

    def initialize_swa(self):
        # Ensure we use the optimizer configured by the parent (FTLTrainer)
        if self.optimizer is None:
             raise RuntimeError("Optimizer not initialized before SWA setup.")

        network_for_swa = self.network.module if hasattr(self.network, 'module') else self.network
        self.swa_model = AveragedModel(network_for_swa)
        # Initialize SWALR with the optimizer (which already has the correct LR/WD/Type)
        self.swa_scheduler = SWALR(self.optimizer, swa_lr=self.swa_lr)
        self.print_to_log_file("SWA components initialized.")

    def on_epoch_end(self):
        continue_training = super().on_epoch_end()
        if self.use_swa and self.current_epoch >= self.swa_start_epoch:
            self.update_swa()
        return continue_training

    def update_swa(self):
        network = self.network.module if hasattr(self.network, 'module') else self.network
        self.swa_model.update_parameters(network)
        self.swa_scheduler.step()
        self.current_lr = self.optimizer.param_groups[0]['lr']

    def on_train_end(self):
        if self.use_swa and self.current_epoch >= self.swa_start_epoch:
            self.finalize_swa()
        super().on_train_end()

    def finalize_swa(self):
        self.print_to_log_file("Finalizing training: Copying SWA averaged weights.")
        network = self.network.module if hasattr(self.network, 'module') else self.network
        try:
            network.load_state_dict(self.swa_model.state_dict())
            self.print_to_log_file("Updating Normalization layer statistics for SWA model...")
            self.update_norm_stats(network)
            self.print_to_log_file("SWA finalization complete.")
        except Exception as e:
            self.print_to_log_file(f"ERROR during SWA finalization. Error: {e}")

    def update_norm_stats(self, model):
        # Helper function to update BN/IN stats
        requires_update = False
        for module in model.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.InstanceNorm3d)):
                 if hasattr(module, 'track_running_stats') and module.track_running_stats:
                     requires_update = True; break
        
        if not requires_update: return

        model.train(); momenta = {}
        for module in model.modules():
            if isinstance(module, (torch.nn.modules.batchnorm._BatchNorm, torch.nn.InstanceNorm3d)):
                if hasattr(module, 'track_running_stats') and module.track_running_stats:
                    if hasattr(module, 'momentum'):
                        momenta[module] = module.momentum; module.momentum = None
                    if hasattr(module, 'reset_running_stats'):
                        module.reset_running_stats()

        data_loader = self.dataloader_train
        try:
            num_batches = 0
            for data_dict in data_loader:
                data = data_dict['data']; data = data.to(self.device, non_blocking=True)
                with torch.no_grad(): model(data)
                num_batches += 1
                if num_batches >= 250: break
        except Exception as e:
            self.print_to_log_file(f"WARNING: Issue during Norm stats update. Error: {e}")
        
        for module, momentum in momenta.items():
            module.momentum = momentum
