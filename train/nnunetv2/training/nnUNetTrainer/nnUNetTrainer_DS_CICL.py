# Filename: nnUNetTrainer_DS_CICL.py
import os
import sys
import numpy as np
import torch

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


class nnUNetTrainer_DS_CICL(nnUNetTrainer):
    """
    Combines Dynamic Sampling (Lesion specific) with Clinically-Informed Curriculum Learning.
    """
    # Explicit __init__ signature to prevent introspection errors (KeyError: 'args')
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)

        # ========================== CONFIGURATION ==========================
        
        # --- Dynamic Sampling (DS) Configuration ---
        self.num_negative_samples_limit = 1000 # Max negatives allowed in an epoch
        self.target_label_name = "pancreatic_lesion"
        self.target_label_id = self.find_label_id(self.target_label_name)
        
        # --- Curriculum Learning (CICL) Configuration ---
        self.initial_subset_fraction = 0.20  # Start with 20%
        # Reach full dataset utilization by 60% of the training duration
        self.full_dataset_epoch = int(self.num_epochs * 0.6)
        self.pacing_shape = 2.0 # Polynomial pacing shape (P)

        # --- Strategy Setting ---
        # Set to True for Hardest first (Anti-Curriculum), False for Easiest first (Standard Curriculum)
        # !!! IMPORTANT: Configure the desired strategy here !!!
        self.anti_curriculum = False 
        # ===================================================================

        # Placeholders
        self.dl_tr_raw = None
        self.dl_val_raw = None
        self.difficulty_scores = None
        self.sorted_identifiers = None # Sorted list of all training identifiers
        self.classification = {} # Dictionary mapping identifiers to 'positive' or 'negative'
        self.previous_epoch_indices = []
        self.enable_framework = True

        # Validation and Logging
        if self.target_label_id is None:
            self.print_to_log_file(f"ERROR: Target label '{self.target_label_name}' not found. Disabling framework.")
            self.enable_framework = False

        strategy_name = "Anti-Curriculum (Hardest First)" if self.anti_curriculum else "Curriculum (Easiest First)"
        self.print_to_log_file("Initialized nnUNetTrainer_DS_CICL (Dynamic Sampling + Curriculum).")
        self.print_to_log_file(f"  Strategy: {strategy_name}. Target Label ID: {self.target_label_id}. Max Negatives: {self.num_negative_samples_limit}.")
        self.print_to_log_file(f"  Pacing: Initial {self.initial_subset_fraction*100:.1f}%. Full data by Epoch: {self.full_dataset_epoch}. Shape (P): {self.pacing_shape}.")

    # ====================================================================================
    # Helper Methods (Data Loader Lifecycle and Utilities)
    # ====================================================================================

    def find_label_id(self, label_name):
        # (Utility to find label ID from dataset.json)
        labels = self.dataset_json.get("labels", {})
        for key, value in labels.items():
            if key == label_name:
                try: return int(value)
                except (ValueError, TypeError): pass
            elif value == label_name:
                try: return int(key)
                except (ValueError, TypeError): pass
        return None

    # (Methods _create_augmenter and _stop_dataloader use robust implementations from previous fixes)
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
    # Combined Strategy Logic (Classification, Sorting, and Selection)
    # ====================================================================================

    def prepare_dataset_strategy(self, dataset):
        """Performs both difficulty sorting (CICL) and positive/negative classification (DS)."""
        if not self.enable_framework:
            return

        # 1. Load Difficulty Scores (CICL) - Robust path resolution
        self.print_to_log_file("Loading difficulty scores...")
        base_dir = self.preprocessed_dataset_folder
        original_dir = base_dir
        while base_dir and not os.path.basename(base_dir).startswith("Dataset"):
             parent_dir = os.path.dirname(base_dir)
             if parent_dir == base_dir: 
                 base_dir = original_dir
                 break
             base_dir = parent_dir
        
        scores_file = join(base_dir, 'difficulty_scores.pkl')

        if not isfile(scores_file):
            self.print_to_log_file(f"ERROR: difficulty_scores.pkl not found at {scores_file}. Disabling framework.")
            self.enable_framework = False
            return

        self.difficulty_scores = load_pickle(scores_file)

        # 2. Classify Keys (DS)
        self.print_to_log_file(f"Classifying keys based on label ID: {self.target_label_id}...")
        self.classification = {}
        
        for identifier in dataset.identifiers:
            properties_file = join(dataset.source_folder, identifier + '.pkl')
            if not isfile(properties_file):
                 self.classification[identifier] = 'negative' # Assume negative if properties missing
                 continue

            properties = load_pickle(properties_file)
            class_locations = properties.get('class_locations', {})

            is_positive = False
            if self.target_label_id in class_locations:
                if len(class_locations[self.target_label_id]) > 0:
                    is_positive = True
            
            self.classification[identifier] = 'positive' if is_positive else 'negative'

        # 3. Align and Sort (Integration)
        scores_for_split = {}
        for identifier in dataset.identifiers:
            # Use default score 0.0 if missing
            scores_for_split[identifier] = self.difficulty_scores.get(identifier, 0.0)

        # Sort based on strategy (reverse=True for Anti-Curriculum)
        self.sorted_identifiers = sorted(scores_for_split, key=scores_for_split.get, reverse=self.anti_curriculum)
        
        # Summary Logging
        total_pos = sum(1 for k in self.classification if self.classification[k] == 'positive')
        total_neg = len(self.classification) - total_pos
        self.print_to_log_file(f"Preparation complete. Total Positive: {total_pos}, Total Negative: {total_neg}.")


    def calculate_pacing_fraction(self):
        # (Polynomial Pacing function logic)
        if self.current_epoch >= self.full_dataset_epoch:
            return 1.0
        progress = self.current_epoch / self.full_dataset_epoch
        shaped_progress = progress ** self.pacing_shape
        fraction = self.initial_subset_fraction + (1.0 - self.initial_subset_fraction) * shaped_progress
        return min(fraction, 1.0)

    def select_keys_for_epoch(self):
        """Combines CICL pacing with DS balancing."""
        if not self.enable_framework or self.sorted_identifiers is None:
            return

        # Step 1: Determine Curriculum Subset (CICL Pacing)
        fraction = self.calculate_pacing_fraction()
        total_cases = len(self.sorted_identifiers)
        num_cases_for_curriculum = int(np.ceil(total_cases * fraction))
        
        if num_cases_for_curriculum == 0:
             print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} Selection: 0 cases.")
             if self.dl_tr_raw: self.dl_tr_raw.indices = []
             return

        # Select the subset based on the sorted list (Easiest or Hardest N cases)
        curriculum_subset = self.sorted_identifiers[:num_cases_for_curriculum]

        # Step 2: Apply Dynamic Sampling (DS Balancing) within the subset
        
        # Separate the subset into positive and negative lists
        positive_keys_subset = [k for k in curriculum_subset if self.classification[k] == 'positive']
        negative_keys_subset = [k for k in curriculum_subset if self.classification[k] == 'negative']
        
        # Start with all positives in the subset
        selected_keys = list(positive_keys_subset)
        
        # Select a random subset of the available negatives up to the limit
        num_negative_to_select = min(self.num_negative_samples_limit, len(negative_keys_subset))

        # Use a deterministic RNG for reproducibility (DDP safe).
        rng = np.random.RandomState(self.current_epoch + self.local_rank)
        
        if num_negative_to_select > 0:
            selected_negative_keys = rng.choice(negative_keys_subset, num_negative_to_select, replace=False).tolist()
            selected_keys.extend(selected_negative_keys)
        else:
            selected_negative_keys = []

        # Shuffle the final selection
        shuffled_keys = list(selected_keys)
        rng.shuffle(shuffled_keys)

        # Logging (Detailed breakdown)
        boundary_case_id = self.sorted_identifiers[num_cases_for_curriculum-1]
        boundary_difficulty = self.difficulty_scores.get(boundary_case_id, -1)
        boundary_desc = "Min Difficulty" if self.anti_curriculum else "Max Difficulty"
        
        # Use print for visibility across DDP ranks
        print(f"[Rank {self.local_rank}] Epoch {self.current_epoch} Selection:")
        print(f"  Curriculum Subset: {len(curriculum_subset)}/{total_cases} cases ({fraction*100:.1f}%). {boundary_desc}: {boundary_difficulty:.4f}.")
        print(f"  Dynamic Sampling: {len(positive_keys_subset)} Pos + {len(selected_negative_keys)} Neg. Total Selected: {len(shuffled_keys)}.")

        # Update the raw dataloader indices
        if self.dl_tr_raw:
            self.dl_tr_raw.indices = shuffled_keys

    # ====================================================================================
    # Overridden Methods (Lifecycle management)
    # ====================================================================================
    
    # (The lifecycle methods utilize the robust infrastructure developed in previous iterations 
    # to handle dynamic worker restarting.)

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # (Configuration setup...)
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

        # --- Combined Strategy Implementation: Preparation ---
        self.prepare_dataset_strategy(dataset_tr)
        # -----------------------------------------------------

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

        # Signal dynamic management
        return None, None

    def on_train_start(self):
        super().on_train_start()
        if self.dl_val_raw:
            self.print_to_log_file("Starting validation augmenter...")
            self.dataloader_val = self._create_augmenter(self.dl_val_raw, is_train=False)
            try: _ = next(self.dataloader_val)
            except StopIteration: pass

    def on_train_epoch_start(self):
        # 1. Select keys (Combined strategy)
        if self.enable_framework:
            self.select_keys_for_epoch()
        
        # 2. Check if the subset has changed (Optimization)
        subset_changed = True
        
        if self.dl_tr_raw is not None:
            current_indices = self.dl_tr_raw.indices
            # Compare sorted lists as order is randomized each epoch.
            if sorted(self.previous_epoch_indices) == sorted(current_indices):
                 subset_changed = False
            self.previous_epoch_indices = list(current_indices)

        # 3. Restart the training augmenter if necessary
        if subset_changed or self.dataloader_train is None:
            if self.dataloader_train is not None:
                self._stop_dataloader(self.dataloader_train)
            
            if self.dl_tr_raw:
                 self.dataloader_train = self._create_augmenter(self.dl_tr_raw, is_train=True)
                 try: 
                     _ = next(self.dataloader_train)
                 except StopIteration: 
                     if self.enable_framework and len(self.dl_tr_raw.indices) == 0:
                          self.print_to_log_file("Note: Strategy selected 0 cases for this epoch.")
                     else:
                        self.print_to_log_file("Warning: Training dataloader startup failed.")
            else:
                 raise RuntimeError("Error: Raw dataloader missing.")

        # (Standard epoch start logic)
        self.network.train()
        if hasattr(self, 'lr_scheduler') and self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)
            
        self.print_to_log_file(f'\nEpoch {self.current_epoch}')
        
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.print_to_log_file(f"Current learning rate: {np.round(lr, decimals=5)}")
            if hasattr(self, 'logger') and self.logger is not None:
                 self.logger.log('lrs', lr, self.current_epoch)

    def on_train_end(self):
        # (Standard cleanup logic)
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
