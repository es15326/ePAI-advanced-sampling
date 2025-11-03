import os
import sys
import numpy as np
import torch

# Imports from batchgenerators
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_pickle

# Imports from nnunetv2

# --- Relative Import Strategy ---
# This is necessary if the file is placed inside nnunetv2/training/nnUNetTrainer
# to resolve import ambiguities when running via the nnUNetv2_train command line tool.
try:
    # The dot (.) signifies the current directory/package.
    from .nnUNetTrainer import nnUNetTrainer
except ImportError as e:
    print(f"WARNING: Relative import failed: {e}. Attempting fallback imports.")
    try:
        # Fallback for standard recent versions if relative import fails
        from nnunetv2.training.trainer.nnUNetTrainer import nnUNetTrainer
    except ImportError:
        # Handle cases where the file might be imported from an external extension setup
        try:
            from nnunetv2.training.nnUNetTrainer import nnUNetTrainer
        except ImportError:
             print("ERROR: All import methods failed. Please check your nnunetv2 installation structure and trainer placement.")
             raise
# --- End Relative Import Strategy ---


from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainer_DynamicSamplingLesion(nnUNetTrainer):
    """
    Custom nnUNetTrainer for dynamic sampling strategy focused on a specific label.
    """
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Custom Configuration ---
        self.num_negative_samples_per_epoch = 200
        self.enable_custom_sampling = True

        # Define the target label for positive classification
        self.target_label_name = "pancreatic_lesion"
        # Automatically find the ID from dataset.json
        self.target_label_id = self.find_label_id(self.target_label_name)
        # ----------------------------

        # Placeholders for the restructured data loading
        self.dl_tr_raw = None
        self.dl_val_raw = None
        self.positive_keys_tr = None
        self.negative_keys_tr = None
        self.annotated_classes_key_tr = None # Kept for compatibility

        if self.target_label_id is not None:
            self.print_to_log_file(f"Initialized nnUNetTrainer_DynamicSampling.")
            self.print_to_log_file(f"Target Label: '{self.target_label_name}' (ID: {self.target_label_id}). Target negative samples: {self.num_negative_samples_per_epoch}")
        else:
            self.print_to_log_file(f"WARNING: Target label '{self.target_label_name}' not found in dataset.json. Disabling custom sampling.")
            self.enable_custom_sampling = False

    # ====================================================================================
    # Helper Methods
    # ====================================================================================

    def find_label_id(self, label_name):
        """Finds the integer ID for a given label name in dataset.json (Robustly)."""
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            # Check standard structure (e.g., "pancreatic_lesion": 18)
            if key == label_name:
                if isinstance(value, int):
                    return value
                elif isinstance(value, str) and value.isdigit():
                    return int(value)
            # Check reversed structure (e.g., "18": "pancreatic_lesion")
            elif value == label_name:
                 if isinstance(key, int):
                    return key
                 elif isinstance(key, str) and key.isdigit():
                    return int(key)
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

            # Use seeds based on the epoch and rank for reproducibility when restarting workers.
            base_seed = self.current_epoch * 1000 + self.local_rank * num_processes
            seeds = [base_seed + i for i in range(num_processes)]

            return NonDetMultiThreadedAugmenter(data_loader=dataloader, transform=None,
                                                num_processes=num_processes,
                                                num_cached=num_cached, seeds=seeds,
                                                pin_memory=self.device.type == 'cuda', wait_time=0.002)

    def _stop_dataloader(self, dataloader):
        """Utility to stop the augmenter gracefully."""
        if dataloader is not None:
            # Suppress stdout messages about stopping workers, as this happens every epoch.
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
            # Should be caught by __init__, but safety check here as well.
            return dataset.identifiers, []

        positive_keys = []
        negative_keys = []

        for identifier in dataset.identifiers:
            properties_file = join(dataset.source_folder, identifier + '.pkl')
            if not isfile(properties_file):
                 self.print_to_log_file(f"Warning: Properties file not found for {identifier}. Skipping.")
                 continue

            # Load the preprocessed properties file
            properties = load_pickle(properties_file)
            # 'class_locations' stores coordinates of labels found during preprocessing.
            class_locations = properties.get('class_locations', {})

            # --- MODIFIED LOGIC: Check specifically for the target label ---
            is_positive = False
            # Check if the target label ID exists in the class locations AND has entries (voxels).
            if self.target_label_id in class_locations:
                if len(class_locations[self.target_label_id]) > 0:
                    is_positive = True
            # ----------------------------------------------------------------

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

        # Start with all positive samples
        selected_keys = list(self.positive_keys_tr)
        num_positive = len(selected_keys)

        # Select a subset of negative samples
        num_negative_to_select = min(self.num_negative_samples_per_epoch, len(self.negative_keys_tr))

        # Use a deterministic RNG based on the epoch and rank for reproducibility (DDP safe).
        rng = np.random.RandomState(self.current_epoch + self.local_rank)

        if num_negative_to_select > 0:
            selected_negative_keys = rng.choice(self.negative_keys_tr, num_negative_to_select, replace=False).tolist()
            selected_keys.extend(selected_negative_keys)
        else:
            selected_negative_keys = []

        # Shuffle the combined list
        rng.shuffle(selected_keys)

        # Log the selection (using print so all DDP ranks report their selection)
        print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} selection: {num_positive} positive ({self.target_label_name}), {len(selected_negative_keys)} negative. Total {len(selected_keys)} cases.")

        # Update the raw dataloader indices on the current worker
        if self.dl_tr_raw:
            self.dl_tr_raw.indices = selected_keys
        else:
            self.print_to_log_file("Error: Raw training dataloader (dl_tr_raw) not available.")

    # ====================================================================================
    # Overridden Methods (Lifecycle management - unchanged from previous implementation)
    # ====================================================================================

    def get_dataloaders(self):
        """
        Modified to initialize raw data loaders and perform classification, but not start the augmenters.
        """
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # (Setup configuration: patch size, transforms, etc. - Copied from base trainer)
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

        # Initialize the nnUNetDataLoaders
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

        # Store raw dataloaders
        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        # Classify keys if custom sampling is enabled
        if self.enable_custom_sampling:
            # Store this key if needed for other logic, though not strictly required for targeted classification
            if hasattr(dl_tr, 'annotated_classes_key'):
                self.annotated_classes_key_tr = dl_tr.annotated_classes_key

            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_tr)
            self.print_to_log_file(f"Training set analysis: {len(self.positive_keys_tr)} positive cases ({self.target_label_name}), {len(self.negative_keys_tr)} negative cases.")

            if len(self.negative_keys_tr) < self.num_negative_samples_per_epoch:
                self.print_to_log_file(f"INFO: Total negative samples ({len(self.negative_keys_tr)}) is less than the requested subset ({self.num_negative_samples_per_epoch}). Using all.")

        # We do not create the MultiThreadedAugmenters here. Return None.
        return None, None

    def on_train_start(self):
        """
        Modified to utilize the overridden get_dataloaders and manually start the validation augmenter.
        """
        # 1. Call the parent implementation. This calls our overridden get_dataloaders().
        super().on_train_start()

        # 2. Manually start the validation augmenter (static dataset).
        if self.dl_val_raw:
            self.print_to_log_file("Starting validation augmenter...")
            # Overwrite the None set by the parent method
            self.dataloader_val = self._create_augmenter(self.dl_val_raw, is_train=False)
            # Start the workers by requesting the first batch
            try:
                _ = next(self.dataloader_val)
            except StopIteration:
                self.print_to_log_file("Validation dataloader is empty.")

    def on_train_epoch_start(self):
        """
        Implements the dynamic resampling and worker restart logic.
        """
        # --- Dynamic Sampling Implementation ---

        if self.enable_custom_sampling:
            # 1. Stop previous training augmenter (if it exists) to clear the prefetch queue.
            if self.dataloader_train is not None:
                self._stop_dataloader(self.dataloader_train)

            # 2. Select keys for this epoch (updates self.dl_tr_raw.indices)
            self.select_keys_for_epoch()

            # 3. Create and start a new training augmenter
            if self.dl_tr_raw:
                self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
                # Start the workers
                try:
                    _ = next(self.dataloader_train)
                except StopIteration:
                    self.print_to_log_file("Warning: Training dataloader startup failed (StopIteration). Check if selected keys are valid.")
            else:
                raise RuntimeError("Error: Cannot start training augmenter, raw dataloader missing.")

        # Handle the case where custom sampling is disabled (fallback to standard behavior)
        elif self.dataloader_train is None and self.dl_tr_raw is not None:
             self.print_to_log_file("Starting training augmenter (default sampling)...")
             self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
             try:
                 _ = next(self.dataloader_train)
             except StopIteration:
                 self.print_to_log_file("Warning: Training dataloader startup failed.")

        # (Original logic follows: network mode, LR scheduler, logging)
        self.network.train()
        self.lr_scheduler.step(self.current_epoch)
        self.print_to_log_file('')
        self.print_to_log_file(f'Epoch {self.current_epoch}')
        self.print_to_log_file(
            f"Current learning rate: {np.round(self.optimizer.param_groups[0]['lr'], decimals=5)}")
        self.logger.log('lrs', self.optimizer.param_groups[0]['lr'], self.current_epoch)

    def on_train_end(self):
        """
        Modified to use the _stop_dataloader utility for cleanup.
        """
        # (Logic copied from base nnUNetTrainer.on_train_end)
        self.current_epoch -= 1
        self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        self.current_epoch += 1

        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            os.remove(join(self.output_folder, "checkpoint_latest.pth"))

        # shut down dataloaders using the utility function.
        self._stop_dataloader(self.dataloader_train)
        self._stop_dataloader(self.dataloader_val)

        empty_cache(self.device)
        self.print_to_log_file("Training done.")
