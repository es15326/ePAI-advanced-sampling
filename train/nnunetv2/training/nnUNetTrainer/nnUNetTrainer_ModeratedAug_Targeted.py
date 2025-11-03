# Filename: nnUNetTrainer_ModeratedAug_Targeted.py
import torch
import numpy as np
import os # Import os module

# (Standard Imports...)
try:
    from .nnUNetTrainer import nnUNetTrainer
except ImportError:
    try:
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("ERROR: Cannot import base nnUNetTrainer."); raise

try:
    from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted
except ImportError:
    print("ERROR: Cannot import nnUNetDataLoader_Targeted."); raise

# (Other imports: DataLoader, Augmenter, Transforms...)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
# (Importing others for type checking and default moderated settings)
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

class nnUNetTrainer_ModeratedAug_Targeted(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        self.target_label_name = "pancreatic_lesion"
        self.target_label_id = self.find_label_id(self.target_label_name)
        
        # Set defaults
        self.oversample_foreground_percent = 0.50
        self.target_sampling_ratio = 0.75
        self.aug_scale_min = 0.7
        self.aug_spatial_p = 0.25

        # --- HYPERPARAMETER TUNING VIA ENVIRONMENT VARIABLES (Augmentation & Sampling) ---
        try:
            # Sampling Strategy
            self.oversample_foreground_percent = float(os.environ.get("OVERSAMPLE_FG", 0.50))
            self.target_sampling_ratio = float(os.environ.get("TARGET_RATIO", 0.75))
            
            # Augmentation Intensity
            self.aug_scale_min = float(os.environ.get("AUG_SCALE_MIN", 0.7))
            self.aug_spatial_p = float(os.environ.get("AUG_SPATIAL_P", 0.25))
            
            # Log overrides
            if any(k in os.environ for k in ["OVERSAMPLE_FG", "TARGET_RATIO", "AUG_SCALE_MIN", "AUG_SPATIAL_P"]):
                 self.print_to_log_file("INFO: Augmentation/Sampling Hyperparameters overridden by environment variables.")

        except ValueError as e:
            self.print_to_log_file(f"ERROR parsing HParam environment variables. Using defaults. Error: {e}")
        
        # Calculate derived parameters and validate
        if self.aug_scale_min >= 1.0 or self.aug_scale_min <= 0.0:
            self.print_to_log_file(f"WARNING: AUG_SCALE_MIN ({self.aug_scale_min}) must be between 0.0 and 1.0. Resetting to 0.7.")
            self.aug_scale_min = 0.7
        self.aug_scale_max = 1.0 / self.aug_scale_min # Maintain symmetry
        # -------------------------------------------------------

        self.print_to_log_file("Initialized nnUNetTrainer_ModeratedAug_Targeted.")
        self.print_to_log_file(f"  Sampling Config: Foreground Pct={self.oversample_foreground_percent:.2f}, Target Ratio={self.target_sampling_ratio:.2f}.")
        self.print_to_log_file(f"  Augmentation Config: Scale Range=({self.aug_scale_min:.2f}, {self.aug_scale_max:.2f}), Spatial P={self.aug_spatial_p:.2f}.")

    # (find_label_id implementation)
    def find_label_id(self, label_name):
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            if key == label_name:
                try: return int(value)
                except: pass
            elif value == label_name:
                try: return int(key)
                except: pass
        return None

    # Override to implement Tunable Moderated Augmentation
    def get_training_transforms(self, *args, **kwargs) -> AbstractTransform:
        
        base_transforms = super().get_training_transforms(*args, **kwargs)
        
        if hasattr(base_transforms, 'transforms') and len(base_transforms.transforms) > 0:
            for transform in base_transforms.transforms:
                
                # Spatial Transformations
                if isinstance(transform, SpatialTransform):
                    # Use the Tunable Probability
                    transform.p_rot_per_sample = self.aug_spatial_p
                    transform.p_scale_per_sample = self.aug_spatial_p
                    transform.p_el_per_sample = self.aug_spatial_p
                    
                    # Use the Tunable Scale Range
                    transform.scale = (self.aug_scale_min, self.aug_scale_max)

                    # Keep elastic deformation magnitudes moderated
                    transform.elastic_deform_alpha = (0., 1000.)
                    transform.elastic_deform_sigma = (9., 13.)
                    
                # (Other moderated augmentations remain unchanged from the baseline)
                elif isinstance(transform, GammaTransform):
                    transform.p_per_sample = 0.2; transform.gamma_range = (0.6, 1.8)
                elif isinstance(transform, (BrightnessMultiplicativeTransform, ContrastAugmentationTransform)):
                     transform.p_per_sample = 0.2
                elif isinstance(transform, GaussianNoiseTransform):
                    transform.p_per_sample = 0.15
                elif isinstance(transform, GaussianBlurTransform):
                    transform.p_per_sample = 0.25
                elif isinstance(transform, SimulateLowResolutionTransform):
                    transform.p_per_sample = 0.3

        self.print_to_log_file("Applied Tunable Moderated Augmentation Parameters.")
        return base_transforms

    # Override to use the Tunable Sampling Ratios
    def get_dataloaders(self):
        # (Standard setup logic... identical to previous implementations)
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

        # --- MODIFICATION: Use Targeted DataLoader with Tunable Parameters ---
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
            self.label_manager,
            # Use the Tunable Hyperparameters
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms,
            target_label_id=self.target_label_id,
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                    self.configuration_manager.patch_size, self.label_manager,
                                    oversample_foreground_percent=self.oversample_foreground_percent,
                                    transforms=val_transforms)
        # ------------------------------------------------------------------------

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
