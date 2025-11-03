# Filename: nnUNetTrainer_Targeted.py
import torch
import numpy as np # Required for type hints in DataLoader

# (Robust Relative Import Strategy for nnUNetTrainer)
try:
    from .nnUNetTrainer import nnUNetTrainer
except ImportError:
    try:
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("ERROR: Cannot import base nnUNetTrainer."); raise

# Import the custom DataLoader
try:
    from .nnUNetDataLoader_Targeted import nnUNetDataLoader_Targeted
except ImportError:
     print("ERROR: Cannot import nnUNetDataLoader_Targeted."); raise

# Import standard components required for get_dataloaders
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA


class nnUNetTrainer_Targeted(nnUNetTrainer):
    # FIX: Explicit __init__ signature to avoid introspection errors (KeyError: 'args')
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- Configuration ---
        self.target_label_name = "pancreatic_lesion"
        self.target_label_id = self.find_label_id(self.target_label_name)
        self.target_sampling_ratio = 0.75 # 75% of foreground patches will prioritize the lesion
        
        # Intense Foreground Oversampling (66% of all patches are foreground)
        self.oversample_foreground_percent = 0.66 
        # ---------------------

        self.print_to_log_file(f"Initialized nnUNetTrainer_Targeted. Target ID: {self.target_label_id}. Ratio: {self.target_sampling_ratio}. Foreground Pct: {self.oversample_foreground_percent}.")

    def find_label_id(self, label_name):
        # Helper to find the label ID from dataset.json
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            if key == label_name:
                try: return int(value)
                except (ValueError, TypeError): pass
            elif value == label_name:
                try: return int(key)
                except (ValueError, TypeError): pass
        return None

    def get_dataloaders(self):
        # Override to use the custom nnUNetDataLoader_Targeted
        
        # (Standard setup logic...)
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

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

        # --- MODIFICATION: Use Targeted DataLoader for Training ---
        # We pass the arguments required by the constructor.
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr, 
            self.batch_size,
            initial_patch_size, # initial_patch_size
            self.configuration_manager.patch_size, # final_patch_size
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms,
            # Custom arguments:
            target_label_id=self.target_label_id,
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        # Use the standard loader for validation.
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  transforms=val_transforms)
        # ----------------------------------------------------------

        # (Standard augmenter creation logic...)
        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            return SingleThreadedAugmenter(dl_tr, None), SingleThreadedAugmenter(dl_val, None)
        else:
            tr_processes = allowed_num_processes
            tr_cached = max(6, allowed_num_processes // 2)
            
            mt_gen_train = NonDetMultiThreadedAugmenter(dl_tr, None, tr_processes, tr_cached,
                                                        pin_memory=self.device.type == 'cuda')
            mt_gen_val = NonDetMultiThreadedAugmenter(dl_val, None, max(1, tr_processes // 2), max(3, tr_cached // 2),
                                                      pin_memory=self.device.type == 'cuda')
            return mt_gen_train, mt_gen_val
