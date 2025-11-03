# Filename: nnUNetTrainer_DS_TO.py
import torch
import numpy as np

# --- FIX: Update imports based on user-specified filenames and use aliases for clarity ---

# Import the base Dynamic Sampling trainer (using relative import)
# We assume the class name inside 'nnUNetTrainer_DynamicSampling_targeted.py' is 'nnUNetTrainer_DynamicSampling'.
# If you also renamed the class inside the file, please update the import statement accordingly.
try:
    # Updated filename: nnUNetTrainer_DynamicSampling_targeted.py
    from .nnUNetTrainer_DynamicSampling_targeted import nnUNetTrainer_DynamicSampling_targeted as nnUNetTrainer_DynamicSampling
except ImportError:
     print("ERROR: Cannot import nnUNetTrainer_DynamicSampling from nnUNetTrainer_DynamicSampling_targeted.py. Check filename, location, and class name."); raise

# Import the custom Targeted DataLoader (using relative import)
# We assume the class name inside 'nnUNetDataLoader_Targeted_dynamic.py' is 'nnUNetDataLoader_Targeted'.
try:
    # Updated filename: nnUNetDataLoader_Targeted_dynamic.py
    # We alias it for clarity within this file.
    # from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted_dynamic as nnUNetDataLoader_Targeted
    from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted
except ImportError:
    print("ERROR: Cannot import nnUNetDataLoader_Targeted from nnUNetDataLoader_Targeted_dynamic.py. Check filename, location, and class name."); raise
# ------------------------------------------------------------------------------------------

# Import standard components required for get_dataloaders
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader # Needed for validation loader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

class nnUNetTrainer_DS_TO(nnUNetTrainer_DynamicSampling):
    """
    Combines Dynamic Sampling (Case Level) with Targeted Oversampling (Patch Level).
    """
    # Explicit __init__ signature to prevent introspection errors
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        
        # Initialize the base Dynamic Sampling trainer.
        # This sets up DS parameters (target_label_id, num_negative_samples_limit)
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Targeted Oversampling (TO) Configuration ---
        self.target_sampling_ratio = 0.75 
        
        # Increase the overall foreground sampling percentage
        self.oversample_foreground_percent = 0.66 # 66% of patches are foreground
        # ------------------------------------------------

        self.print_to_log_file("Initialized Hybrid nnUNetTrainer_DS_TO (Dynamic Sampling + Targeted Oversampling).")
        self.print_to_log_file(f"  TO Config: Ratio={self.target_sampling_ratio}, Foreground Pct={self.oversample_foreground_percent}.")

    # FIX: Robustly override get_dataloaders to initialize the custom loader directly.
    # This avoids the AttributeError encountered when trying to extract configuration attributes.
    def get_dataloaders(self):
        """
        Manages the initialization of both Targeted (Training) and Standard (Validation) loaders,
        and executes the Dynamic Sampling classification logic.
        """
        
        # 1. Perform standard configuration setup (Replicating logic from base nnUNetTrainer)
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # Configuration setup: patch size, transforms, etc.
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
        # We pass the configuration directly.
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr, # Pass the dataset object directly (matches expected argument 'data' in base DataLoader)
            self.batch_size,
            initial_patch_size, # initial_patch_size
            self.configuration_manager.patch_size, # final_patch_size
            self.label_manager,
            # Use the hybrid trainer's setting for oversampling
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            # Custom arguments for TO:
            # self.target_label_id is inherited from DynamicSampling init
            target_label_id=self.target_label_id, 
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        # 3. Instantiate the Standard DataLoader for Validation
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  transforms=val_transforms, 
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        # 4. Store the raw loaders
        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        # 5. Perform Dynamic Sampling Classification (DS Logic)
        # This logic is required here because we override the parent method completely.
        if self.enable_custom_sampling:
            # Set annotated_classes_key if available
            if hasattr(dl_tr, 'annotated_classes_key'):
                self.annotated_classes_key_tr = dl_tr.annotated_classes_key

            # Call the classification logic (inherited from nnUNetTrainer_DynamicSampling)
            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_tr)
            self.print_to_log_file(f"Training set analysis: {len(self.positive_keys_tr)} positive, {len(self.negative_keys_tr)} negative.")

            # Check if the attribute exists before accessing (robustness)
            if hasattr(self, 'num_negative_samples_per_epoch') and len(self.negative_keys_tr) < self.num_negative_samples_per_epoch:
                 self.print_to_log_file(f"INFO: Total negative samples ({len(self.negative_keys_tr)}) is less than the requested subset ({self.num_negative_samples_per_epoch}). Using all.")

        # Return (None, None) as required by the Dynamic Sampling lifecycle management
        return None, None

    # Note: Lifecycle methods (on_train_epoch_start, etc.) and helper methods (classify_keys, select_keys_for_epoch) 
    # are correctly inherited from nnUNetTrainer_DynamicSampling.
