# Filename: nnUNetTrainer_IntenseAug_Targeted.py
import torch
import numpy as np

# (Robust Relative Import Strategy for nnUNetTrainer)
try:
    from .nnUNetTrainer import nnUNetTrainer
except ImportError:
    try:
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        print("ERROR: Cannot import base nnUNetTrainer."); raise

# Import the custom Targeted DataLoader
try:
    # We assume the class name inside 'nnUNetDataLoader_Targeted_dynamic.py' is 'nnUNetDataLoader_Targeted'.
    from .nnUNetDataLoader_Targeted_dynamic_aug import nnUNetDataLoader_Targeted
except ImportError:
    print("ERROR: Cannot import nnUNetDataLoader_Targeted from nnUNetDataLoader_Targeted_dynamic.py."); raise

# Import standard components required for get_dataloaders and augmentation
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

# Import augmentation transforms for type checking and parameter modification
from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform

class nnUNetTrainer_IntenseAug_Targeted(nnUNetTrainer):
    # Explicit __init__ signature to avoid introspection errors (KeyError: 'args')
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        
        # --- Configuration ---
        self.target_label_name = "pancreatic_lesion"
        self.target_label_id = self.find_label_id(self.target_label_name)
        
        # Targeted Oversampling (TO) Configuration
        self.target_sampling_ratio = 0.75 
        # Intense foreground sampling (mimics --crop_on_tumor emphasis)
        self.oversample_foreground_percent = 0.66 
        # ---------------------

        self.print_to_log_file("Initialized nnUNetTrainer_IntenseAug_Targeted.")
        self.print_to_log_file(f"  Strategy: Intensified Augmentation + Targeted Oversampling (ID: {self.target_label_id}, Ratio: {self.target_sampling_ratio}, Foreground Pct: {self.oversample_foreground_percent}).")

    def find_label_id(self, label_name):
        # (Helper to find label ID)
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            if key == label_name:
                try: return int(value)
                except: pass
            elif value == label_name:
                try: return int(key)
                except: pass
        return None

    # Override to implement Intensified Augmentation
    def get_training_transforms(self, *args, **kwargs) -> AbstractTransform:
        
        # 1. Call the base implementation to get the standard nnU-Net transforms
        # This ensures all necessary components (like deep supervision handling) are included.
        base_transforms = super().get_training_transforms(*args, **kwargs)
        
        # 2. Modify the parameters of the existing transforms for higher intensity.
        # This approach is robust as it modifies the existing structure rather than redefining it.
        
        # The standard transforms are encapsulated in a Compose transform.
        if hasattr(base_transforms, 'transforms') and len(base_transforms.transforms) > 0:
            # Iterate through the transforms within the Compose object
            for transform in base_transforms.transforms:
                
                # Intensify Spatial Transformations
                if isinstance(transform, SpatialTransform):
                    # Increase probability of spatial augmentation
                    transform.p_rot_per_sample = 0.3 # Default 0.2
                    transform.p_scale_per_sample = 0.3 # Default 0.2
                    transform.p_el_per_sample = 0.3 # Default 0.2
                    
                    # Increase the magnitude of elastic deformation.
                    transform.elastic_deform_alpha = (0., 1200.) # Default (0., 900.)
                    transform.elastic_deform_sigma = (10., 15.) # Default (9., 13.)
                    
                    # Increase scaling range
                    transform.scale = (0.65, 1.6) # Default (0.7, 1.4)

                # Intensify Intensity/Color Transformations
                elif isinstance(transform, GammaTransform):
                    transform.p_per_sample = 0.3 # Default 0.15
                    # Wider range for gamma correction
                    transform.gamma_range = (0.5, 2.0) # Default (0.7, 1.5)
                
                elif isinstance(transform, (BrightnessMultiplicativeTransform, ContrastAugmentationTransform)):
                     transform.p_per_sample = 0.3 # Default 0.15

                # Intensify Noise and Blurring
                elif isinstance(transform, GaussianNoiseTransform):
                    transform.p_per_sample = 0.2 # Default 0.15
                
                elif isinstance(transform, GaussianBlurTransform):
                    transform.p_per_sample = 0.3 # Default 0.2

                # Intensify Low Resolution Simulation
                elif isinstance(transform, SimulateLowResolutionTransform):
                    transform.p_per_sample = 0.4 # Default 0.25

            self.print_to_log_file("Applied Intensified Augmentation Parameters.")
        
        return base_transforms

    # Override to use the custom nnUNetDataLoader_Targeted
    def get_dataloaders(self):
        
        # (Standard setup logic...)
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes,) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        
        # Call the overridden get_training_transforms method (Intensified Augmentation)
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
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr, 
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
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
