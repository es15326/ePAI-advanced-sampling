# Filename: nnUNetTrainer_Curriculum_DS_TO.py
import torch
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any

# Import the NEW base DS_TO trainer (nnUNetTrainer_DS_TO_Base.py)
try:
    # We inherit from the robust base implementation.
    from .nnUNetTrainer_DS_TO_Base import nnUNetTrainer_DS_TO_Base as nnUNetTrainer_DS_TO_Foundation
except ImportError:
    print("FATAL: Could not import nnUNetTrainer_DS_TO_Base. Make sure nnUNetTrainer_DS_TO_Base.py is available.")
    # Fallback definition for type hinting/static analysis if the import fails.
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as nnUNetTrainer_DS_TO_Foundation

# --------------------------------------------------------------------------------------------------
# Curriculum Manager Utility (10 Metrics Implementation)
# --------------------------------------------------------------------------------------------------

class CurriculumManager:
    """Handles metadata loading, difficulty scoring, and key ordering based on PanTS_reports.csv."""
    def __init__(self, metadata_csv_path: str):
        self.metadata_csv_path = metadata_csv_path
        self.scores: pd.DataFrame = pd.DataFrame()
        self.strategy_metrics: List[str] = []
        self.is_initialized = self._calculate_all_scores()

    def _calculate_all_scores(self):
        """Calculates 10 different hardness metrics based on the metadata."""
        if not os.path.exists(self.metadata_csv_path):
            return False

        try:
            df = pd.read_csv(self.metadata_csv_path)
            df.columns = df.columns.str.strip()
            if 'PanTS ID' not in df.columns:
                print("CurriculumManager: 'PanTS ID' column missing.")
                return False
            df['PanTS ID'] = df['PanTS ID'].astype(str)
            df.set_index('PanTS ID', inplace=True)
        except Exception as e:
            print(f"CurriculumManager: Failed to load metadata CSV: {e}")
            return False

        # Define Columns
        PANC_VOL = 'pancreas volume (cm^3)'
        LESION_VOL = 'total pancreatic lesion volume (cm^3)'
        TUMOR_VOL = 'total pancreatic tumor volume (cm^3)'
        CYST_VOL = 'total pancreatic cyst volume (cm^3)'
        NUM_LESIONS = 'number of pancreatic lesion instances'
        LESION_DIAMETER = 'largest pancreatic lesion diameter (cm)'
        LOCATION = 'largest pancreatic lesion location (head, body, tail)'
        ATTENUATION = 'largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)'
        NARRATIVE = 'narrative report'

        # Clean numeric columns
        numeric_cols = [PANC_VOL, LESION_VOL, TUMOR_VOL, CYST_VOL, NUM_LESIONS, LESION_DIAMETER]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

        scores = pd.DataFrame(index=df.index)

        # --- Metric Implementations (Higher Score = Harder Case) ---

        # S1: Inverse Lesion Volume
        scores['S1_Inv_Volume'] = 1 / (df.get(LESION_VOL, 0) + 1e-6)

        # S2: Inverse Lesion-to-Pancreas Ratio
        ratio = df.get(LESION_VOL, 0) / (df.get(PANC_VOL, 0) + 1e-6)
        scores['S2_Inv_Ratio'] = 1 / (ratio + 1e-6)

        # S3: Instance Complexity
        scores['S3_Instances'] = df.get(NUM_LESIONS, 0)

        # S4: Attenuation Difficulty
        def map_attenuation(val):
            if isinstance(val, str):
                val = val.lower()
                if 'isoattenuating' in val: return 5
                if 'hypoattenuating' in val: return 3
                if 'hyperattenuating' in val: return 1
            return 3
        scores['S4_Attenuation'] = df[ATTENUATION].apply(map_attenuation) if ATTENUATION in df.columns else 3

        # S5: Location Difficulty
        def map_location(val):
            if isinstance(val, str):
                val = val.lower()
                score = 0
                if 'head' in val: score = max(score, 5)
                if 'body' in val: score = max(score, 3)
                if 'tail' in val: score = max(score, 1)
                return score if score > 0 else 3
            return 3
        scores['S5_Location'] = df[LOCATION].apply(map_location) if LOCATION in df.columns else 3

        # S6: Geometric Complexity Proxy (D^3/V)
        complexity = (df.get(LESION_DIAMETER, 0)**3) / (df.get(LESION_VOL, 0) + 1e-6)
        scores['S6_Geometry'] = complexity.replace([np.inf, -np.inf], 0).fillna(0)

        # S7: Mixed Pathology
        has_tumor = df.get(TUMOR_VOL, 0) > 0
        has_cyst = df.get(CYST_VOL, 0) > 0
        scores['S7_Mixed_Pathology'] = (has_tumor & has_cyst).astype(int)

        # S8: Organ Morphology Deviation
        panc_vols = df.get(PANC_VOL, pd.Series(dtype=float))
        median_panc_vol = panc_vols[panc_vols > 0].median()
        if pd.isna(median_panc_vol): median_panc_vol = 0
        scores['S8_Organ_Deviation'] = (panc_vols - median_panc_vol).abs()

        # S9: Boundary Ambiguity Proxy (HU Std Dev from Narrative)
        def extract_hu_sd(report):
            if isinstance(report, str):
                match = re.search(r'\+/-\s*([\d.]+)', report)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        return 0.0
            return 0.0
        scores['S9_HU_StdDev'] = df[NARRATIVE].apply(extract_hu_sd) if NARRATIVE in df.columns else 0

        # S10: Composite Score (Rank Aggregation)
        valid_scores = scores.dropna(axis=1, how='all')
        ranks = valid_scores.rank(method='average', ascending=True)
        scores['S10_Composite'] = ranks.sum(axis=1)

        self.scores = scores
        self.strategy_metrics = scores.columns.tolist()
        return True

    def get_ordered_keys(self, metric_name: str, positive_keys: List[str], reverse: bool) -> List[str]:
        """
        Orders keys. reverse=False (E2H), reverse=True (H2E).
        """
        if metric_name not in self.strategy_metrics or self.scores.empty:
            keys_copy = list(positive_keys)
            np.random.shuffle(keys_copy)
            return keys_copy

        positive_keys_str = [str(k) for k in positive_keys]
        current_scores = self.scores[metric_name].reindex(positive_keys_str)
        
        if current_scores.isnull().any():
            median_score = current_scores.median()
            if pd.isna(median_score):
                median_score = 0
            current_scores = current_scores.fillna(median_score)

        sorted_series = current_scores.sort_values(ascending=not reverse)
        return sorted_series.index.tolist()

# --------------------------------------------------------------------------------------------------
# Curriculum Trainer (Implementation relying on the fixed DS_TO_Base)
# --------------------------------------------------------------------------------------------------

class nnUNetTrainer_Curriculum_DS_TO(nnUNetTrainer_DS_TO_Foundation):
    
    def initialize(self):
        
        # --- 1. Robust CSV Path Finding ---
        csv_filename = "PanTS_reports.csv"
        csv_path = None
        search_locations = []

        # Define search locations
        if hasattr(self, 'preprocessed_dataset_folder') and self.preprocessed_dataset_folder:
            search_locations.append(self.preprocessed_dataset_folder) # Config specific folder
            
            parent_dir = os.path.dirname(self.preprocessed_dataset_folder)
            if parent_dir and os.path.isdir(parent_dir) and parent_dir != self.preprocessed_dataset_folder:
                search_locations.append(parent_dir) # Dataset folder

                grandparent_dir = os.path.dirname(parent_dir)
                if grandparent_dir and os.path.isdir(grandparent_dir) and grandparent_dir != parent_dir:
                    search_locations.append(grandparent_dir) # nnUNet_preprocessed root

        # Search the locations
        for loc in search_locations:
            potential_path = os.path.join(loc, csv_filename)
            if os.path.exists(potential_path):
                csv_path = potential_path
                break

        # --- 2. Initialize Curriculum Manager ---
        if csv_path:
             # Use print during initialization as logger might not be ready.
             print(f"INFO: Found metadata CSV at: {csv_path}")
             self.curriculum_manager = CurriculumManager(csv_path)
        else:
             print(f"ERROR: Metadata CSV '{csv_filename}' not found. Searched locations: {search_locations}")
             self.curriculum_manager = CurriculumManager("/invalid/path/for/dummy/manager") 

        self.curriculum_schedule = self._define_schedule()

        # --- 3. Standard Initialization ---
        # This calls the parent's (DS_TO_Base) initialize and get_dataloaders.
        super().initialize()

        # --- 4. Post-Initialization Fixes and Logging ---

        # Ensure RNG is initialized and seeded correctly (Handles 'all' folds)
        if not hasattr(self, 'RNG') or self.RNG is None:
            self.print_to_log_file("WARNING: RNG (Random Number Generator) not found from parent, initializing in Curriculum Trainer.")
            
            fold_val = getattr(self, 'fold', None)
            if isinstance(fold_val, int):
                seed = fold_val
            elif fold_val == 'all' or fold_val is None:
                seed = 0 # Default seed for 'all' folds
                self.print_to_log_file(f"INFO: Fold is '{fold_val}'. Using default seed {seed} for RNG.")
            else:
                try:
                    seed = int(fold_val)
                except (ValueError, TypeError):
                    seed = 42 # Fallback seed
                    self.print_to_log_file(f"WARNING: Unexpected fold value '{fold_val}'. Using fallback seed {seed} for RNG.")
            
            self.RNG = np.random.default_rng(seed)

        # Log initialization status
        if self.curriculum_manager.is_initialized:
            self.print_to_log_file(f"Curriculum Learning initialized. Cycle length: {len(self.curriculum_schedule)} epochs.")
        else:
            self.print_to_log_file("WARNING: CurriculumManager failed to initialize. Curriculum learning will be inactive (fallback to random sampling).")


    def on_train_start(self):
        # CRITICAL FIX: Handle the complex initialization sequence of DS/TO inheritance.

        # --- 1. Bridging (Ensure augmentation setup works) ---
        # We must populate self.dataloader_train/val using the internally stored loaders (self.dl_tr/val) 
        # BEFORE the base on_train_start() sets up the augmenters (tr_gen/val_gen).
        # The fix in the parent (DS_TO_Base) ensures dl_tr/dl_val exist now.

        if self.dataloader_train is None and hasattr(self, 'dl_tr') and self.dl_tr is not None:
             self.print_to_log_file("INFO: Bridging DS/TO initialization: Populating self.dataloader_train from self.dl_tr.")
             self.dataloader_train = self.dl_tr
        
        if self.dataloader_val is None and hasattr(self, 'dl_val') and self.dl_val is not None:
             self.dataloader_val = self.dl_val

        # --- 2. Call Parent Implementation ---
        # This initializes the augmentation pipeline (tr_gen/val_gen).
        super().on_train_start()

        # --- 3. Re-linking (Ensure the main loop iterator is correct) ---
        # If the parent nullified the dataloader (common in DS), we MUST link it back 
        # to the augmenter (tr_gen/val_gen) so the main training loop can iterate.

        if hasattr(self, 'tr_gen') and self.tr_gen is not None:
            # Check if it's None or if it doesn't match the augmenter
            if self.dataloader_train is None or self.dataloader_train != self.tr_gen:
                self.print_to_log_file("INFO: Re-linking self.dataloader_train to self.tr_gen (Augmenter) after parent initialization.")
                self.dataloader_train = self.tr_gen
            
        if hasattr(self, 'val_gen') and self.val_gen is not None:
             if self.dataloader_val is None or self.dataloader_val != self.val_gen:
                 self.dataloader_val = self.val_gen

        # --- 4. Final Safety Check ---
        if self.dataloader_train is None:
            # Provide a robust fallback if augmentation failed but the raw loader exists
            if hasattr(self, 'dl_tr') and self.dl_tr is not None and (not hasattr(self, 'tr_gen') or self.tr_gen is None):
                 self.print_to_log_file("WARNING: Augmentation (tr_gen) initialization seems to have failed. Falling back to raw loader (dl_tr).")
                 self.dataloader_train = self.dl_tr
            
            # If it is still None, we cannot train.
            if self.dataloader_train is None:
                raise RuntimeError(
                    "Training initialization failed. self.dataloader_train is None after on_train_start(). "
                    "Check initialization sequence of dl_tr and tr_gen across the inheritance chain."
                )


    def _define_schedule(self):
        """Defines the sequence of curriculum strategies (E2H and H2E for all metrics)."""
        schedule = []
        if self.curriculum_manager:
            for metric in self.curriculum_manager.strategy_metrics:
                # E2H: Ascending scores (reverse=False)
                schedule.append({'metric': metric, 'order': 'E2H', 'reverse': False})
                # H2E: Descending scores (reverse=True)
                schedule.append({'metric': metric, 'order': 'H2E', 'reverse': True})
        return schedule

    def _update_dataloader_indices(self, keys_for_epoch):
        """
        Helper function to update indices in dl_tr and tr_gen robustly.
        """
        
        def update_loader(loader):
            if loader is None:
                return False
            
            # Pattern 1: Custom DataLoader managing indices directly
            if hasattr(loader, 'indices'):
                loader.indices = keys_for_epoch
                return True
            
            # Pattern 2: Standard PyTorch Subset pattern
            if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'indices'):
                loader.dataset.indices = keys_for_epoch
                return True
                
            return False

        updated_dl_tr = False
        # Update the raw loader stored by the parent (dl_tr)
        if hasattr(self, 'dl_tr'):
            updated_dl_tr = update_loader(self.dl_tr)

        updated_tr_gen = False
        # The augmenter (tr_gen) holds the active loader in .data_loader attribute
        if hasattr(self, 'tr_gen') and self.tr_gen is not None and hasattr(self.tr_gen, 'data_loader'):
             # It's crucial to update the loader that tr_gen is actively using.
             updated_tr_gen = update_loader(self.tr_gen.data_loader)
        
        return updated_dl_tr or updated_tr_gen


    def on_train_epoch_start(self):
        """
        Overrides the Dynamic Sampling logic to implement the curriculum with Ordered Interleaving.
        """
        
        # 1. Standard nnU-Net epoch start procedures.
        if hasattr(self, 'network'):
            self.network.train()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)

        # Check if Dynamic Sampling is enabled
        if not getattr(self, 'enable_custom_sampling', False):
            return

        # Ensure keys and attributes are available
        if not hasattr(self, 'positive_keys_tr') or not hasattr(self, 'negative_keys_tr'):
            # Attempt recovery if keys are missing (e.g. if parent initialization skipped classification)
            # Check if we have the means to classify keys (access to dataset and the classification method)
            if hasattr(self, 'classify_keys'):
                 self.print_to_log_file("INFO: Positive/Negative keys missing. Attempting to run key classification now.")
                 try:
                     # Determine the dataset source robustly
                     dataset_to_use = None
                     if hasattr(self, 'dataset_tr') and self.dataset_tr is not None:
                         dataset_to_use = self.dataset_tr
                     elif hasattr(self, 'dl_tr') and hasattr(self.dl_tr, 'dataset'):
                         dataset_to_use = self.dl_tr.dataset

                     if dataset_to_use:
                        self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_to_use)
                     else:
                         raise ValueError("Cannot access training dataset for key classification.")
                         
                 except Exception as e:
                     self.print_to_log_file(f"ERROR: Failed to classify keys during on_train_epoch_start: {e}")
                     return
            else:
                self.print_to_log_file("ERROR: Missing positive/negative keys and cannot re-classify. Skipping epoch sampling.")
                return
        
        if not hasattr(self, 'negative_to_positive_ratio'):
            self.negative_to_positive_ratio = 1.0

        # --- Curriculum Dynamic Sampling Logic ---
        
        positive_keys = list(self.positive_keys_tr)
        negative_keys = list(self.negative_keys_tr)

        # 2. Apply Curriculum Sorting (Positive keys)
        if self.curriculum_manager and self.curriculum_manager.is_initialized and self.curriculum_schedule:
            strategy_index = self.current_epoch % len(self.curriculum_schedule)
            current_strategy = self.curriculum_schedule[strategy_index]
            
            self.print_to_log_file(f"Epoch {self.current_epoch}: Curriculum Strategy: {current_strategy['metric']} ({current_strategy['order']})")
            
            sorted_positive_keys = self.curriculum_manager.get_ordered_keys(
                current_strategy['metric'], 
                positive_keys,
                reverse=current_strategy['reverse']
            )
        else:
            # Fallback to random shuffle if manager failed
            self.RNG.shuffle(positive_keys) 
            sorted_positive_keys = positive_keys

        # 3. Handle Negative Keys (Standard DS implementation)
        num_pos = len(sorted_positive_keys)
        num_neg_to_sample = int(np.round(num_pos * self.negative_to_positive_ratio))
        
        if num_neg_to_sample == 0 or len(negative_keys) == 0:
             sampled_negative_keys = []
        elif num_neg_to_sample > len(negative_keys):
             # Not enough negatives, repeat as necessary
            sampled_negative_keys = negative_keys * (num_neg_to_sample // len(negative_keys))
            remaining = num_neg_to_sample % len(negative_keys)
            if remaining > 0:
                sampled_negative_keys += list(self.RNG.choice(negative_keys, remaining, replace=False))
        else:
            # Randomly sample the required number of negatives
            sampled_negative_keys = list(self.RNG.choice(negative_keys, num_neg_to_sample, replace=False))
        
        # Shuffle the negative keys
        self.RNG.shuffle(sampled_negative_keys)

        # 4. Ordered Interleaving (Maintains curriculum order while balancing P/N ratio)
        
        keys_for_epoch = []
        neg_ptr = 0
        ratio = self.negative_to_positive_ratio
        
        if num_pos == 0:
             keys_for_epoch = sampled_negative_keys
        else:
            for i, pos_key in enumerate(sorted_positive_keys):
                keys_for_epoch.append(pos_key)
                ideal_neg_count = round((i + 1) * ratio)
                neg_to_insert = ideal_neg_count - neg_ptr
                
                for j in range(neg_to_insert):
                    if neg_ptr < len(sampled_negative_keys):
                        keys_for_epoch.append(sampled_negative_keys[neg_ptr])
                        neg_ptr += 1

            # Append any remaining negatives
            while neg_ptr < len(sampled_negative_keys):
                keys_for_epoch.append(sampled_negative_keys[neg_ptr])
                neg_ptr += 1

        # 5. Update the data loader
        self.print_to_log_file(f"Epoch {self.current_epoch} sampling summary: {len(keys_for_epoch)} cases ({num_pos} pos, {len(sampled_negative_keys)} neg). Interleaved.")
        
        if not self._update_dataloader_indices(keys_for_epoch):
             self.print_to_log_file("CRITICAL ERROR: Could not update data loader indices (dl_tr or tr_gen). Check structure.")
             raise RuntimeError("Failed to configure DataLoader indices for the epoch.")

        # 6. Handle distributed training synchronization (DDP)
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == 0:
                object_list = [keys_for_epoch]
            else:
                object_list = [None]
            
            torch.distributed.broadcast_object_list(object_list, src=0)
            
            if torch.distributed.get_rank() != 0:
                keys_for_epoch = object_list[0]
                if not self._update_dataloader_indices(keys_for_epoch):
                     raise RuntimeError("Failed to configure DataLoader indices on distributed worker.")
