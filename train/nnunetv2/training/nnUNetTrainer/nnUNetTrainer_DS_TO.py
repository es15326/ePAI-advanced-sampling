# Filename: nnUNetTrainer_DS_TO.py
import torch
import numpy as np

# Import the base Dynamic Sampling trainer
try:
    from .nnUNetTrainer_DynamicSampling_targeted import nnUNetTrainer_DynamicSampling_targeted as nnUNetTrainer_DynamicSampling
except ImportError:
    print("ERROR: Cannot import nnUNetTrainer_DynamicSampling from nnUNetTrainer_DynamicSampling_targeted.py."); raise

# Import the custom Targeted DataLoader (Assumes nnUNetDataLoader_Targeted_dynamic.py exists)
try:
    from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted
except ImportError:
    print("ERROR: Cannot import nnUNetDataLoader_Targeted from nnUNetDataLoader_Targeted_dynamic.py."); raise

# Import standard components
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

class nnUNetTrainer_DS_TO(nnUNetTrainer_DynamicSampling):
    """
    Combines Dynamic Sampling (Case Level) with Targeted Oversampling (Patch Level).
    """
    
    # Use **kwargs for flexibility. This is safe as it's an intermediate class.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):
        
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)

        # --- Targeted Oversampling (TO) Configuration ---
        self.target_sampling_ratio = 0.75 
        self.oversample_foreground_percent = 0.66 # 66% of patches are foreground

        # Logging
        print("Initialized Hybrid nnUNetTrainer_DS_TO (Dynamic Sampling + Targeted Oversampling).")
        print(f"  TO Config: Ratio={self.target_sampling_ratio}, Foreground Pct={self.oversample_foreground_percent}.")


    def get_dataloaders(self):
        """
        Manages the initialization of loaders and executes the Dynamic Sampling classification logic.
        """
        
        # 1. Perform standard configuration setup
        if getattr(self, 'dataset_class', None) is None:
             # Ensure preprocessed_dataset_folder is available (it should be after super().__init__)
             if hasattr(self, 'preprocessed_dataset_folder') and self.preprocessed_dataset_folder:
                 self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
             else:
                 # This might happen if the base trainer initialization sequence is non-standard
                 print("WARNING: preprocessed_dataset_folder not yet available during get_dataloaders.")


        # Configuration setup: patch size, transforms, etc.
        # These managers should be initialized by the (patched) base trainer.
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes,) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)
        
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        # 2. Instantiate the Targeted DataLoader for Training (TO Injection)
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr, 
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms,
            probabilistic_oversampling=getattr(self, 'probabilistic_oversampling', False),
            # Pass the target_label_id defined in the parent (DynamicSampling_targeted)
            target_label_id=getattr(self, 'target_label_id', None),
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        # 3. Instantiate the Standard DataLoader for Validation
        if dataset_val is not None:
            dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size, self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        transforms=val_transforms, 
                                        probabilistic_oversampling=getattr(self, 'probabilistic_oversampling', False))
        else:
            dl_val = None

        # 4. Store the raw loaders (Required by this DS/TO structure)
        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        # 5. Perform Dynamic Sampling Classification (DS Logic)
        if getattr(self, 'enable_custom_sampling', False):
            # Call the classification logic
            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_tr)
            
            # Logging
            log_func = getattr(self, 'print_to_log_file', print)
            log_func(f"Training set analysis: {len(self.positive_keys_tr)} positive, {len(self.negative_keys_tr)} negative.")

            if hasattr(self, 'num_negative_samples_per_epoch'):
                    max_neg = self.num_negative_samples_per_epoch
                    if isinstance(max_neg, (int, float)) and not isinstance(max_neg, bool) and len(self.negative_keys_tr) < max_neg:
                        log_func(f"INFO: Total negative samples ({len(self.negative_keys_tr)}) is less than the requested subset ({max_neg}). Using all.")

        # Return (None, None)
        return None, None
