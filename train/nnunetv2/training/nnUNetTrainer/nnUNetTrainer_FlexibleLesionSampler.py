import os
import sys
import numpy as np
import torch
from datetime import datetime
import shutil 

# Imports from batchgenerators
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_pickle, maybe_mkdir_p, save_json

# Imports from nnunetv2
try:
    from .nnUNetTrainer import nnUNetTrainer
except ImportError as e:
    print(f"WARNING: Relative import failed: {e}. Attempting fallback imports.")
    try:
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        try:
            from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
        except ImportError:
             print("ERROR: All import methods failed. Please check your nnunetv2 installation structure and trainer placement.")
             raise

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainer_FlexibleLesionSampler(nnUNetTrainer):
    """
    Flexible custom trainer that reads settings from environment variables
    AND creates a unique output folder for each experiment.
    - NNUNET_NEG_SAMPLES: Number of negative samples per epoch.
    - NNUNET_FG_OVERSAMPLE: The oversample_foreground_percent (patch-level).
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        
        # This call sets the default self.oversample_foreground_percent = 0.5
        # It also sets self.output_folder_base and self.output_folder
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Custom Configuration from Environment Variables ---
        default_neg_samples = 100
        default_fg_oversample = 0.5

        try:
            neg_samples_str = os.environ.get('NNUNET_NEG_SAMPLES', str(default_neg_samples))
            self.num_negative_samples_per_epoch = int(neg_samples_str)
        except ValueError:
            print(f"Warning: Invalid NNUNET_NEG_SAMPLES. Using default: {default_neg_samples}")
            neg_samples_str = str(default_neg_samples)
            self.num_negative_samples_per_epoch = default_neg_samples

        try:
            fg_oversample_str = os.environ.get('NNUNET_FG_OVERSAMPLE', str(default_fg_oversample))
            self.oversample_foreground_percent = float(fg_oversample_str)
        except ValueError:
            print(f"Warning: Invalid NNUNET_FG_OVERSAMPLE. Using default: {default_fg_oversample}")
            fg_oversample_str = str(default_fg_oversample)
            self.oversample_foreground_percent = default_fg_oversample

        # --- !!! NEW: MODIFY OUTPUT FOLDER !!! ---
        fg_oversample_suffix = fg_oversample_str.replace('.', 'p')
        experiment_suffix = f"_fg{fg_oversample_suffix}_neg{neg_samples_str}"
        self.output_folder_base = self.output_folder_base + experiment_suffix
        self.output_folder = join(self.output_folder_base, f'fold_{self.fold}')

        timestamp = datetime.now()
        maybe_mkdir_p(self.output_folder) 
        self.log_file = join(self.output_folder, "training_log_%d_%d_%d_%02.0d_%02.0d_%02.0d.txt" %
                             (timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute,
                              timestamp.second))
        # --- END OF MODIFICATION ---
        
        self.enable_custom_sampling = True
        self.target_label_name = "pancreatic_lesion"
        self.target_label_id = self.find_label_id(self.target_label_name)

        # Placeholders
        self.dl_tr_raw = None
        self.dl_val_raw = None
        self.positive_keys_tr = None
        self.negative_keys_tr = None
        self.annotated_classes_key_tr = None 

        if self.target_label_id is not None:
            self.print_to_log_file("--- FlexibleSampler Configuration ---")
            self.print_to_log_file(f"Target Label: '{self.target_label_name}' (ID: {self.target_label_id})")
            self.print_to_log_file(f"Case-Level Negative Samples: {self.num_negative_samples_per_epoch} (from NNUNET_NEG_SAMPLES)")
            self.print_to_log_file(f"Patch-Level FG Oversample: {self.oversample_foreground_percent} (from NNUNET_FG_OVERSAMPLE)")
            self.print_to_log_file(f"Output folder set to: {self.output_folder}")
            self.print_to_log_file("-------------------------------------")
        else:
            self.print_to_log_file(f"WARNING: Target label '{self.target_label_name}' not found. Disabling custom sampling.")
            self.enable_custom_sampling = False

    # ====================================================================================
    # Helper Methods (NO CHANGES)
    # ====================================================================================

    def find_label_id(self, label_name):
        """Finds the integer ID for a given label name in dataset.json (Robustly)."""
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            if key == label_name:
                if isinstance(value, int): return value
                elif isinstance(value, str) and value.isdigit(): return int(value)
            elif value == label_name:
                 if isinstance(key, int): return key
                 elif isinstance(key, str) and key.isdigit(): return int(key)
        return None

    def _create_augmenter(self, dataloader, is_train=True):
        """Utility to create SingleThreaded or NonDetMultiThreadedAugmenter."""
        allowed_num_processes = get_allowed_n_proc_DA()

        if allowed_num_processes == 0:
            return SingleThreadedAugmenter(dataloader, None)
        else:
            if is_train:
                num_processes = allowed_num_processes
                num_cached = max(6, allowed_num_processes // 2)
            else:
                num_processes = max(1, allowed_num_processes // 2)
                num_cached = max(3, allowed_num_processes // 4)
            base_seed = self.current_epoch * 1000 + self.local_rank * num_processes
            seeds = [base_seed + i for i in range(num_processes)]
            return NonDetMultiThreadedAugmenter(data_loader=dataloader, transform=None,
                                                num_processes=num_processes,
                                                num_cached=num_cached, seeds=seeds,
                                                pin_memory=self.device.type == 'cuda', wait_time=0.002)

    def _stop_dataloader(self, dataloader):
        """Utility to stop the augmenter gracefully."""
        if dataloader is not None:
            old_stdout = sys.stdout
            try:
                with open(os.devnull, 'w') as f:
                    sys.stdout = f
                    if isinstance(dataloader, (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter)):
                        if hasattr(dataloader, '_finish'):
                            dataloader._finish()
            except Exception as e:
                sys.stdout = old_stdout
                self.print_to_log_file(f"Error stopping dataloader: {e}")
            finally:
                sys.stdout = old_stdout

    def classify_keys(self, dataset):
        """
        Classifies keys based on the presence of the specific target_label_id.
        """
        self.print_to_log_file(f"Classifying dataset keys based on label ID: {self.target_label_id}...")
        if self.target_label_id is None:
            return dataset.identifiers, []

        positive_keys = []
        negative_keys = []
        for identifier in dataset.identifiers:
            properties_file = join(dataset.source_folder, identifier + '.pkl')
            if not isfile(properties_file):
                 self.print_to_log_file(f"Warning: Properties file not found for {identifier}. Skipping.")
                 continue
            properties = load_pickle(properties_file)
            class_locations = properties.get('class_locations', {})
            is_positive = False
            if self.target_label_id in class_locations:
                if len(class_locations[self.target_label_id]) > 0:
                    is_positive = True
            if is_positive:
                positive_keys.append(identifier)
            else:
                negative_keys.append(identifier)
        self.print_to_log_file("Classification complete.")
        return positive_keys, negative_keys

    def select_keys_for_epoch(self):
        """Selects keys for the epoch based on the custom strategy."""
        if self.positive_keys_tr is None or self.negative_keys_tr is None:
            return

        selected_keys = list(self.positive_keys_tr)
        num_positive = len(selected_keys)
        num_negative_to_select = min(self.num_negative_samples_per_epoch, len(self.negative_keys_tr))
        rng = np.random.RandomState(self.current_epoch + self.local_rank)
        if num_negative_to_select > 0:
            selected_negative_keys = rng.choice(self.negative_keys_tr, num_negative_to_select, replace=False).tolist()
            selected_keys.extend(selected_negative_keys)
        else:
            selected_negative_keys = []
        rng.shuffle(selected_keys)
        print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} selection: {num_positive} positive ({self.target_label_name}), {len(selected_negative_keys)} negative. Total {len(selected_keys)} cases.")
        if self.dl_tr_raw:
            self.dl_tr_raw.indices = selected_keys
        else:
            self.print_to_log_file("Error: Raw training dataloader (dl_tr_raw) not available.")

    # ====================================================================================
    # Overridden Methods (Lifecycle management - CHANGES in on_train_start)
    # ====================================================================================

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)
            
        # --- Standard nnU-Net dataloader setup ---
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

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

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size,
                                 initial_patch_size,
                                 self.configuration_manager.patch_size,
                                 self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 sampling_probabilities=None, pad_sides=None, transforms=tr_transforms,
                                 probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size,
                                  self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size,
                                  self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  sampling_probabilities=None, pad_sides=None, transforms=val_transforms,
                                  probabilistic_oversampling=self.probabilistic_oversampling)
        
        # --- Custom Implementation ---
        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        if self.enable_custom_sampling:
            if hasattr(dl_tr, 'annotated_classes_key'):
                self.annotated_classes_key_tr = dl_tr.annotated_classes_key
            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_tr)
            self.print_to_log_file(f"Training set analysis: {len(self.positive_keys_tr)} positive cases ({self.target_label_name}), {len(self.negative_keys_tr)} negative cases.")
            if len(self.negative_keys_tr) < self.num_negative_samples_per_epoch:
                self.print_to_log_file(f"INFO: Total negative samples ({len(self.negative_keys_tr)}) is less than the requested subset ({self.num_negative_samples_per_epoch}). Using all.")

        return None, None # Return None to prevent default augmenter creation

    def on_train_start(self):
        # --- !!! THIS METHOD IS NOW FIXED (v4) !!! ---

        # 1. Call super().on_train_start() FIRST.
        # This calls our overridden get_dataloaders(), which sets:
        # - self.dl_tr_raw
        # - self.dl_val_raw
        # It also sets self.dataloader_train = None and self.dataloader_val = None
        super().on_train_start()

        # 2. Now, we manually create the validation dataloader (which is static)
        # This was the block I missed in v3.
        if self.dl_val_raw:
            self.print_to_log_file("Starting validation augmenter...")
            self.dataloader_val = self._create_augmenter(self.dl_val_raw, is_train=False)
            try:
                _ = next(self.dataloader_val)
            except StopIteration:
                self.print_to_log_file("Validation dataloader is empty.")

        # 3. Copy the plans and other files to our UNIQUE output folder.
        # We do this *after* super().on_train_start()
        if self.local_rank == 0:
            # We have to check if they exist first, in case super() already copied them
            # (which it shouldn't if we don't modify the path *until* __init__...
            # but this is safer)
            plans_file = join(self.output_folder_base, 'plans.json')
            if not isfile(plans_file):
                save_json(self.plans_manager.plans, plans_file, sort_keys=False)

            dataset_json_file = join(self.output_folder_base, 'dataset.json')
            if not isfile(dataset_json_file):
                save_json(self.dataset_json, dataset_json_file, sort_keys=False)
            
            fingerprint_file = join(self.output_folder_base, 'dataset_fingerprint.json')
            if not isfile(fingerprint_file):
                shutil.copy(join(self.preprocessed_dataset_folder_base, 'dataset_fingerprint.json'),
                            fingerprint_file)
        
        # self.dataloader_train remains None and will be created in on_train_epoch_start
 

    def on_train_epoch_start(self):
        if self.enable_custom_sampling:
            if self.dataloader_train is not None:
                self._stop_dataloader(self.dataloader_train)
            self.select_keys_for_epoch() # This updates self.dl_tr_raw.indices
            if self.dl_tr_raw:
                self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
                try:
                    _ = next(self.dataloader_train)
                except StopIteration:
                    self.print_to_log_file("Warning: Training dataloader startup failed (StopIteration).")
            else:
                raise RuntimeError("Error: Cannot start training augmenter, raw dataloader missing.")
        elif self.dataloader_train is None and self.dl_tr_raw is not None:
             self.print_to_log_file("Starting training augmenter (default sampling)...")
             self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
             try:
                 _ = next(self.dataloader_train)
             except StopIteration:
                 self.print_to_log_file("Warning: Training dataloader startup failed.")

        # (Original logic)
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_end(self):
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1
        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))
        self._stop_dataloader(self.dataloader_train)
        self._stop_dataloader(self.dataloader_val)
        empty_cache(self.device)
        self.print_to_log_file("Training done.")


