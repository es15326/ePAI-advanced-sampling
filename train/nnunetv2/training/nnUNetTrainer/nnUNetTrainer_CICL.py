# Filename: nnUNetTrainer_CICL.py
import os
import sys
import numpy as np
import torch
import math

# Imports from batchgenerators
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, isfile, load_pickle

# --- Relative Import Strategy (Required due to placement constraints) ---
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
             print("ERROR: All import methods failed."); raise
# --------------------------------------------------------------------------

from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainer_CICL(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Curriculum Configuration (Tunable Parameters) ---
        self.enable_curriculum = True
        self.initial_subset_fraction = 0.20  # Start with the easiest 20%
        
        # Epoch at which the full dataset is used (e.g., 60% of total duration)
        self.full_dataset_epoch = int(self.num_epochs * 0.6) 
        
        # Pacing shape (P): Controls the curve (Polynomial Pacing).
        self.pacing_shape = 2.0 
        # -----------------------------------------------------

        # Placeholders
        self.dl_tr_raw = None
        self.dl_val_raw = None
        self.difficulty_scores = None
        self.sorted_identifiers = None # Easiest first

        # FIX: Tracker for indices used in the previous epoch (Resolves AttributeError robustly)
        self.previous_epoch_indices = []

        self.print_to_log_file("Initialized nnUNetTrainer_CICL.")
        self.print_to_log_file(f"  Initial Subset: {self.initial_subset_fraction*100:.1f}%. Full data by Epoch: {self.full_dataset_epoch}. Pacing Shape (P): {self.pacing_shape}.")

    # ====================================================================================
    # Helper Methods (Data Loader Lifecycle Management)
    # ====================================================================================

    # (_create_augmenter and _stop_dataloader remain unchanged)
    def _create_augmenter(self, dataloader, is_train=True):
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

    # ====================================================================================
    # Curriculum Logic
    # ====================================================================================

    # FIX: Updated path resolution (Resolves 'File Not Found' error)
    def load_and_sort_by_difficulty(self, dataset):
        """Loads difficulty scores and sorts the training identifiers."""
        self.print_to_log_file("Loading and sorting dataset by difficulty...")
        
        # Robustly locate the root preprocessed dataset folder (DatasetXXX)
        # self.preprocessed_dataset_folder often points to a configuration subfolder.
        base_dir = self.preprocessed_dataset_folder
        original_dir = base_dir
        
        # Navigate upwards until the DatasetXXX folder is found
        while base_dir and not os.path.basename(base_dir).startswith("Dataset"):
             parent_dir = os.path.dirname(base_dir)
             if parent_dir == base_dir: 
                 # If search fails (e.g., reached filesystem root), fall back to the original path
                 base_dir = original_dir
                 break
             base_dir = parent_dir
        
        scores_file = join(base_dir, 'difficulty_scores.pkl')

        if not isfile(scores_file):
            self.print_to_log_file(f"ERROR: difficulty_scores.pkl not found at {scores_file}. Please verify preprocessing output. Disabling curriculum.")
            self.enable_curriculum = False
            return

        self.difficulty_scores = load_pickle(scores_file)
        
        # Align scores with the current training split (fold)
        training_identifiers = dataset.identifiers
        scores_for_split = {}
        missing_scores = 0

        for identifier in training_identifiers:
            # Assign default difficulty (0.0 - easy) if score is missing
            scores_for_split[identifier] = self.difficulty_scores.get(identifier, 0.0)
            if identifier not in self.difficulty_scores:
                missing_scores += 1

        if missing_scores > 0:
            self.print_to_log_file(f"Note: Missing difficulty scores for {missing_scores} cases. Assigned default (0.0).")

        # Sort identifiers based on scores (ascending order: easiest first)
        self.sorted_identifiers = sorted(scores_for_split, key=scores_for_split.get)
        self.print_to_log_file("Sorting complete.")

    # (calculate_pacing_fraction and select_keys_for_epoch remain unchanged)
    def calculate_pacing_fraction(self):
        if self.current_epoch >= self.full_dataset_epoch:
            return 1.0
        progress = self.current_epoch / self.full_dataset_epoch
        shaped_progress = progress ** self.pacing_shape
        fraction = self.initial_subset_fraction + (1.0 - self.initial_subset_fraction) * shaped_progress
        return min(fraction, 1.0)

    def select_keys_for_epoch(self):
        if not self.enable_curriculum or self.sorted_identifiers is None:
            return

        fraction = self.calculate_pacing_fraction()
        total_cases = len(self.sorted_identifiers)
        num_cases_to_include = int(np.ceil(total_cases * fraction))
        
        if num_cases_to_include == 0:
             print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} Curriculum Selection: 0 cases.")
             if self.dl_tr_raw: self.dl_tr_raw.indices = []
             return

        selected_keys = self.sorted_identifiers[:num_cases_to_include]
        
        rng = np.random.RandomState(self.current_epoch + self.local_rank)
        shuffled_keys = list(selected_keys)
        rng.shuffle(shuffled_keys)

        # Logging
        hardest_case_id = self.sorted_identifiers[num_cases_to_include-1]
        max_difficulty_included = self.difficulty_scores.get(hardest_case_id, -1)
        print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} Curriculum Selection: {len(selected_keys)}/{total_cases} cases ({fraction*100:.1f}%). Max Difficulty: {max_difficulty_included:.4f}.")

        # Update the raw dataloader indices
        if self.dl_tr_raw:
            self.dl_tr_raw.indices = shuffled_keys

    # ====================================================================================
    # Overridden Methods (Lifecycle management)
    # ====================================================================================

    # (get_dataloaders and on_train_start remain unchanged)
    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # (Configuration setup...)
        patch_size = self.configuration_manager.patch_size
        deep_supervision_scales = self._get_deep_supervision_scales()
        (
            rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes,
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

        # --- CICL Implementation: Load and Sort ---
        if self.enable_curriculum:
            self.load_and_sort_by_difficulty(dataset_tr)
        # ------------------------------------------

        dl_tr = nnUNetDataLoader(dataset_tr, self.batch_size, initial_patch_size,
                                 self.configuration_manager.patch_size, self.label_manager,
                                 oversample_foreground_percent=self.oversample_foreground_percent,
                                 transforms=tr_transforms, probabilistic_oversampling=self.probabilistic_oversampling)
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  transforms=val_transforms, probabilistic_oversampling=self.probabilistic_oversampling)

        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        return None, None

    def on_train_start(self):
        super().on_train_start()
        if self.dl_val_raw:
            self.print_to_log_file("Starting validation augmenter...")
            self.dataloader_val = self._create_augmenter(self.dl_val_raw, is_train=False)
            try: _ = next(self.dataloader_val)
            except StopIteration: pass

    # FIX: Revised logic to prevent AttributeError
    def on_train_epoch_start(self):
        """Implements the curriculum selection and worker restart logic."""
        
        # 1. Select keys based on the curriculum (updates self.dl_tr_raw.indices)
        if self.enable_curriculum:
             self.select_keys_for_epoch()
        
        # 2. Check if the subset has changed (Optimization)
        subset_changed = True
        
        if self.dl_tr_raw is not None:
            current_indices = self.dl_tr_raw.indices
            # Compare the current selection with the previous epoch's selection (stored internally)
            # Use sorted comparison as the order is randomized each epoch.
            if sorted(self.previous_epoch_indices) == sorted(current_indices):
                 subset_changed = False
            
            # Update the tracker for the next epoch
            self.previous_epoch_indices = list(current_indices) # Store a copy

        # 3. Restart the training augmenter if necessary
        # We must restart if the subset changed OR if the dataloader hasn't started yet (Epoch 0).
        if subset_changed or self.dataloader_train is None:
            if self.dataloader_train is not None:
                self._stop_dataloader(self.dataloader_train)
            
            if self.dl_tr_raw:
                 self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
                 try: 
                     _ = next(self.dataloader_train)
                 except StopIteration: 
                     # This can happen if the curriculum selects 0 cases
                     if self.enable_curriculum and len(self.dl_tr_raw.indices) == 0:
                          self.print_to_log_file("Note: Curriculum selected 0 cases for this epoch.")
                     else:
                        self.print_to_log_file("Warning: Training dataloader startup failed.")
            else:
                 raise RuntimeError("Error: Raw dataloader missing.")
        # else:
        #     # Optional: Uncomment to confirm optimization is working
        #     # print(f"[Rank {self.local_rank}] Optimization: Subset unchanged, skipping worker restart.")

        # (Standard epoch start logic: network mode, LR scheduler, logging)
        self.network.train()
        # Robust checks for scheduler and optimizer existence
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)
            
        self.print_to_log_file(f'\nEpoch {self.current_epoch}')
        
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.print_to_log_file(f"Current learning rate: {np.round(lr, decimals=5)}")
            if hasattr(self, 'logger') and self.logger is not None:
                 self.logger.log('lrs', lr, self.current_epoch)

    # (on_train_end remains unchanged)
    def on_train_end(self):
        self.current_epoch -= 1
        try: self.save_checkpoint(join(self.output_folder, "checkpoint_final.pth"))
        except Exception: pass
        self.current_epoch += 1

        if self.local_rank == 0 and isfile(join(self.output_folder, "checkpoint_latest.pth")):
            try: os.remove(join(self.output_folder, "checkpoint_latest.pth"))
            except OSError: pass

        self._stop_dataloader(self.dataloader_train)
        self._stop_dataloader(self.dataloader_val)
        empty_cache(self.device)
        self.print_to_log_file("Training done.")
