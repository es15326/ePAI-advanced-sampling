# Filename: nnUNetTrainer_DS_TO_Base.py
import torch
from typing import List
import numpy as np

# Import the base Dynamic Sampling trainer
# Note: Update these imports based on the actual filenames in your environment if they differ.
try:
    # Assuming the DS trainer filename is nnUNetTrainer_DynamicSampling_targeted.py
    from .nnUNetTrainer_DynamicSampling_targeted import nnUNetTrainer_DynamicSampling_targeted as nnUNetTrainer_DynamicSampling
except ImportError:
     print("FATAL: Could not import nnUNetTrainer_DynamicSampling_targeted. Make sure the file is available.")
     # Fallback for type hinting if import fails
     from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as nnUNetTrainer_DynamicSampling

# Import the custom Targeted DataLoader
try:
    # Assuming the DataLoader filename is nnUNetDataLoader_Targeted_dynamic.py
    from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted
except ImportError:
    print("FATAL: Could not import nnUNetDataLoader_Targeted_dynamic. Make sure the file is available.")
    # Fallback for type hinting if import fails
    from nnunetv2.training.data_augmentation.custom_data_augmentation.nnUNetDataLoader import nnUNetDataLoader as nnUNetDataLoader_Targeted

from nnunetv2.training.data_augmentation.custom_data_augmentation.nnUNetDataLoader import nnUNetDataLoader
from nnunetv2.utilities.helpers import dummy_context

class nnUNetTrainer_DS_TO_Base(nnUNetTrainer_DynamicSampling):
    """
    A robust base trainer combining Dynamic Sampling (DS) and Targeted Oversampling (TO).
    Ensures correct initialization sequence for downstream inheritance (e.g., Curriculum Learning).
    """
    
    # Adjust unpack_dataset=True if necessary based on the parent's signature.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        
        # Initialize the base Dynamic Sampling trainer.
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)

        # --- Targeted Oversampling (TO) Configuration ---
        self.target_sampling_ratio = 0.75 
        
        # Increase the overall foreground sampling percentage (used by the DataLoader)
        self.oversample_foreground_percent = 0.66 # 66% of patches are foreground
        
        # These attributes will hold the loaders
        self.dl_tr = None
        self.dl_val = None

        # Attempt logging if the logger is ready
        if hasattr(self, 'print_to_log_file'):
            self.print_to_log_file("Initialized nnUNetTrainer_DS_TO_Base (Robust DS+TO Foundation).")
            self.print_to_log_file(f"  TO Config: Ratio={self.target_sampling_ratio}, Foreground Pct={self.oversample_foreground_percent}.")
        else:
            print("Initialized nnUNetTrainer_DS_TO_Base (Robust DS+TO Foundation).")
            print(f"  TO Config: Ratio={self.target_sampling_ratio}, Foreground Pct={self.oversample_foreground_percent}.")


    def get_dataloaders(self):
        """
        Overrides the base method to inject the Targeted DataLoader and execute DS classification.
        Crucially stores the loaders in self.dl_tr/dl_val before returning (None, None).
        """
        # 1. Standard setup (replicated from base nnUNetTrainer)
        # This initializes self.dataset_tr and self.dataset_val
        self.register_data()

        if self.dataset_tr is None:
             if hasattr(self, 'print_to_log_file'):
                self.print_to_log_file("INFO: Training dataset is empty. Skipping dataloader creation.")
             return None, None

        # Determine patch/batch sizes
        patch_size = self.configuration_manager.patch_size
        batch_size = self.configuration_manager.batch_size
        
        # 2. Instantiate the Targeted DataLoader for Training (TO Injection)
        dl_tr = nnUNetDataLoader_Targeted(
            self.dataset_tr,
            batch_size,
            patch_size,
            self.configuration_manager,
            # Use the hybrid trainer's setting for oversampling
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None, 
            pad_sides=None,
            # Custom arguments for TO:
            target_label_id=getattr(self, 'target_label_id', None), # Inherited from DS init
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        # 3. Instantiate the Standard DataLoader for Validation
        if self.dataset_val is not None:
            dl_val = nnUNetDataLoader(
                self.dataset_val,
                batch_size,
                patch_size,
                self.configuration_manager,
                # Validation uses standard foreground sampling (default 0.33 if attribute missing)
                oversample_foreground_percent=getattr(self, 'oversample_foreground_percent_val', 0.33), 
                sampling_probabilities=None, 
                pad_sides=None
            )
        else:
            dl_val = None

        # 4. CRITICAL FIX: Store the raw loaders internally.
        self.dl_tr = dl_tr
        self.dl_val = dl_val

        # 5. Perform Dynamic Sampling Classification (DS Logic)
        if getattr(self, 'enable_custom_sampling', False):
            
            # Ensure RNG is initialized if not done by the parent
            if not hasattr(self, 'RNG') or self.RNG is None:
                 # Handle 'all' folds for seeding
                 fold_val = getattr(self, 'fold', 0)
                 seed = fold_val if isinstance(fold_val, int) else 0
                 self.RNG = np.random.default_rng(seed)

            target_id = getattr(self, 'target_label_id', 'N/A')
            if hasattr(self, 'print_to_log_file'):
                self.print_to_log_file(f"Classifying dataset keys based on label ID: {target_id}...")
            
            # Call the classification logic (inherited from nnUNetTrainer_DynamicSampling)
            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(self.dataset_tr)
            
            if hasattr(self, 'print_to_log_file'):
                self.print_to_log_file(f"Training set analysis: {len(self.positive_keys_tr)} positive, {len(self.negative_keys_tr)} negative.")

        # 6. Return (None, None) as required by the Dynamic Sampling lifecycle management
        return None, None
