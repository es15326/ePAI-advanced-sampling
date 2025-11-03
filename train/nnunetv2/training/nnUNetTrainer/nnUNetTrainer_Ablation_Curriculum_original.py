# Filename: nnUNetTrainer_Ablation_Curriculum.py
import torch
import numpy as np
import pandas as pd
import os
import re
from typing import List, Dict, Any, Optional

# Import the foundational DS_TO trainer provided by the user (nnUNetTrainer_DS_TO.py)
try:
    # We must inherit from the exact implementation the user provided as the baseline.
    # Assuming the filename is nnUNetTrainer_DS_TO.py
    from .nnUNetTrainer_DS_TO import nnUNetTrainer_DS_TO as nnUNetTrainer_DS_TO_Foundation
except ImportError:
    print("FATAL: Could not import nnUNetTrainer_DS_TO. Make sure nnUNetTrainer_DS_TO.py is available.")
    # Fallback definition for type hinting/static analysis if the import fails.
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer as nnUNetTrainer_DS_TO_Foundation

# --------------------------------------------------------------------------------------------------
# Curriculum Manager Utility (Implementation remains the same)
# --------------------------------------------------------------------------------------------------

class CurriculumManagerAblation:
    """Handles metadata loading, difficulty scoring, and advanced key ordering/filtering."""
    def __init__(self, metadata_csv_path: str):
        self.metadata_csv_path = metadata_csv_path
        self.scores: pd.DataFrame = pd.DataFrame()
        self.strategy_metrics: List[str] = []
        self.is_initialized = self._calculate_all_scores()

    # _calculate_all_scores implementation (Identical to previous robust versions)
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

        # Define Columns and clean data
        PANC_VOL = 'pancreas volume (cm^3)'
        LESION_VOL = 'total pancreatic lesion volume (cm^3)'
        TUMOR_VOL = 'total pancreatic tumor volume (cm^3)'
        CYST_VOL = 'total pancreatic cyst volume (cm^3)'
        NUM_LESIONS = 'number of pancreatic lesion instances'
        LESION_DIAMETER = 'largest pancreatic lesion diameter (cm)'
        LOCATION = 'largest pancreatic lesion location (head, body, tail)'
        ATTENUATION = 'largest pancreatic lesion attenuation (hyperattenuating, isoattenuating, hypoattenuating)'
        NARRATIVE = 'narrative report'

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
        # Ensure there are scores to rank
        if valid_scores.empty:
             scores['S10_Composite'] = 0
        else:
            ranks = valid_scores.rank(method='average', ascending=True)
            scores['S10_Composite'] = ranks.sum(axis=1)

        self.scores = scores
        self.strategy_metrics = scores.columns.tolist()
        return True

    # Helper functions for ordering (Identical to previous versions)
    def _get_scores_for_keys(self, metric_name: str, keys: List[str]):
        if metric_name not in self.strategy_metrics or self.scores.empty:
            return None
        keys_str = [str(k) for k in keys]
        current_scores = self.scores[metric_name].reindex(keys_str)
        if current_scores.isnull().any():
            median_score = current_scores.median()
            if pd.isna(median_score): median_score = 0
            current_scores = current_scores.fillna(median_score)
        return current_scores

    def get_ordered_keys(self, metric_name: str, positive_keys: List[str], order: str) -> List[str]:
        current_scores = self._get_scores_for_keys(metric_name, positive_keys)
        if current_scores is None:
             keys_copy = list(positive_keys)
             np.random.shuffle(keys_copy)
             return keys_copy

        if order == 'E2H':
            sorted_series = current_scores.sort_values(ascending=True)
        elif order == 'H2E':
            sorted_series = current_scores.sort_values(ascending=False)
        elif order in ['MEDIUM_FIRST', 'EXTREMES_FIRST']:
            median_difficulty = current_scores.median()
            deviation = (current_scores - median_difficulty).abs()
            if order == 'MEDIUM_FIRST':
                sorted_series = deviation.sort_values(ascending=True)
            else: 
                sorted_series = deviation.sort_values(ascending=False)
        else:
            raise ValueError(f"Unknown ordering strategy: {order}")
        return sorted_series.index.tolist()

# --------------------------------------------------------------------------------------------------
# Curriculum Trainer (Ablation Implementation - ADAPTED to User's DS_TO)
# --------------------------------------------------------------------------------------------------

class nnUNetTrainer_Ablation_Curriculum(nnUNetTrainer_DS_TO_Foundation):
        
    def initialize(self):
        
        # --- 1. Initialize Curriculum Manager (Path Finding) ---
        csv_filename = "PanTS_reports.csv"
        csv_path = None
        search_locations = []

        if hasattr(self, 'preprocessed_dataset_folder') and self.preprocessed_dataset_folder:
            search_locations.append(self.preprocessed_dataset_folder)
            parent_dir = os.path.dirname(self.preprocessed_dataset_folder)
            if parent_dir and os.path.isdir(parent_dir):
                search_locations.append(parent_dir)
                grandparent_dir = os.path.dirname(parent_dir)
                if grandparent_dir and os.path.isdir(grandparent_dir):
                    search_locations.append(grandparent_dir)

        for loc in search_locations:
            potential_path = os.path.join(loc, csv_filename)
            if os.path.exists(potential_path):
                csv_path = potential_path
                break

        # Initialize Manager
        if csv_path:
             print(f"INFO: Found metadata CSV at: {csv_path}")
             self.curriculum_manager = CurriculumManagerAblation(csv_path)
        else:
             print(f"ERROR: Metadata CSV '{csv_filename}' not found. Searched locations: {search_locations}")
             self.curriculum_manager = CurriculumManagerAblation("/invalid/path/for/dummy/manager") 

        # --- 2. Configure Strategy (Ablation Studies Setup) ---
        self._configure_ablation_strategy()

        # --- 3. Standard Initialization ---
        super().initialize()

        # --- 4. Post-Initialization Fixes and Logging ---

        # Ensure RNG is initialized and seeded correctly
        if not hasattr(self, 'RNG') or self.RNG is None:
            self.print_to_log_file("WARNING: RNG not found from parent, initializing.")
            fold_val = getattr(self, 'fold', None)
            seed = 0
            if isinstance(fold_val, int):
                seed = fold_val
            elif fold_val != 'all' and fold_val is not None:
                 try:
                    seed = int(fold_val)
                 except (ValueError, TypeError):
                    seed = 42
            self.RNG = np.random.default_rng(seed)

        # Log initialization status
        if self.curriculum_manager.is_initialized:
            self.print_to_log_file(f"Curriculum Learning initialized. Strategy: {self.strategy_name}, Metric: {self.strategy_metric}")
        else:
            self.print_to_log_file("WARNING: CurriculumManager failed. Strategy forced to RANDOM.")
            self.strategy_name = "RANDOM"

    def _configure_ablation_strategy(self):
        # (Configuration logic remains the same)
        VALID_STRATEGIES = [
            "CYCLING", "RANDOM", 
            "E2H", "H2E", "MEDIUM_FIRST", "EXTREMES_FIRST", 
            "HARDEST_PCT", "EASIEST_PCT", 
            "STEPPED_E2H", "MEDIUM_OUT"
        ]
        
        self.strategy_name = os.environ.get("CURRICULUM_STRATEGY", "CYCLING").upper()
        self.strategy_metric = os.environ.get("CURRICULUM_METRIC", "S10_Composite")
        
        try:
            self.strategy_percentage = float(os.environ.get("CURRICULUM_PERCENTAGE", "0.5"))
            if not (0 < self.strategy_percentage <= 1.0): self.strategy_percentage = 0.5
        except ValueError:
            self.strategy_percentage = 0.5

        try:
            self.strategy_steps = int(os.environ.get("CURRICULUM_STEPS", "4"))
            if self.strategy_steps < 1: self.strategy_steps = 4
        except ValueError:
            self.strategy_steps = 4

        # Validation
        if self.strategy_name not in VALID_STRATEGIES:
            print(f"WARNING: Invalid CURRICULUM_STRATEGY '{self.strategy_name}'. Defaulting to CYCLING.")
            self.strategy_name = "CYCLING"

        # Check metric validity
        if self.curriculum_manager.is_initialized:
             if self.strategy_name != "CYCLING" and self.strategy_metric not in self.curriculum_manager.strategy_metrics:
                  print(f"WARNING: Invalid CURRICULUM_METRIC '{self.strategy_metric}'. Attempting fallback.")
                  if "S10_Composite" in self.curriculum_manager.strategy_metrics:
                      self.strategy_metric = "S10_Composite"
                  elif self.curriculum_manager.strategy_metrics:
                       self.strategy_metric = self.curriculum_manager.strategy_metrics[0]
                  else:
                       print("ERROR: No metrics available. Forcing RANDOM strategy.")
                       self.strategy_name = "RANDOM"
                       self.strategy_metric = "N/A"
        
        # Define the schedule only if using CYCLING
        self.curriculum_schedule = []
        if self.strategy_name == "CYCLING":
            if self.curriculum_manager.is_initialized:
                for metric in self.curriculum_manager.strategy_metrics:
                    self.curriculum_schedule.append({'metric': metric, 'order': 'E2H'})
                    self.curriculum_schedule.append({'metric': metric, 'order': 'H2E'})

        print(f"INFO: Curriculum Config - Strategy: {self.strategy_name}, Metric: {self.strategy_metric}, Pct: {self.strategy_percentage}, Steps: {self.strategy_steps}")


    def on_train_start(self):
        # ADAPTATION: Handle the initialization sequence specific to the user's nnUNetTrainer_DS_TO.

        # --- 1. Bridging ---
        # The user's implementation uses 'dl_tr_raw' and 'dl_val_raw'. We adapt to these names.
        
        # Check for 'dl_tr_raw' first (as used in the provided file)
        if self.dataloader_train is None:
            if hasattr(self, 'dl_tr_raw') and self.dl_tr_raw is not None:
                 self.print_to_log_file("INFO: Bridging DS/TO initialization: Populating self.dataloader_train from self.dl_tr_raw.")
                 self.dataloader_train = self.dl_tr_raw
            # Fallback to 'dl_tr' if 'dl_tr_raw' is missing (for robustness)
            elif hasattr(self, 'dl_tr') and self.dl_tr is not None:
                 self.print_to_log_file("INFO: Bridging DS/TO initialization: Populating self.dataloader_train from self.dl_tr.")
                 self.dataloader_train = self.dl_tr

        if self.dataloader_val is None:
            if hasattr(self, 'dl_val_raw') and self.dl_val_raw is not None:
                 self.dataloader_val = self.dl_val_raw
            elif hasattr(self, 'dl_val') and self.dl_val is not None:
                 self.dataloader_val = self.dl_val

        # --- 2. Call Parent Implementation ---
        super().on_train_start()

        # --- 3. Re-linking ---
        if hasattr(self, 'tr_gen') and self.tr_gen is not None:
            if self.dataloader_train is None or self.dataloader_train != self.tr_gen:
                self.print_to_log_file("INFO: Re-linking self.dataloader_train to self.tr_gen (Augmenter).")
                self.dataloader_train = self.tr_gen
            
        if hasattr(self, 'val_gen') and self.val_gen is not None:
             if self.dataloader_val is None or self.dataloader_val != self.val_gen:
                 self.dataloader_val = self.val_gen

        # --- 4. Final Safety Check ---
        if self.dataloader_train is None:
            # Fallback logic if augmentation failed
            raw_loader = getattr(self, 'dl_tr_raw', getattr(self, 'dl_tr', None))
            if raw_loader is not None and (not hasattr(self, 'tr_gen') or self.tr_gen is None):
                 self.print_to_log_file("WARNING: Augmentation (tr_gen) failed. Falling back to raw loader.")
                 self.dataloader_train = raw_loader
            
            if self.dataloader_train is None:
                raise RuntimeError(
                    "Training initialization failed. self.dataloader_train is None after on_train_start()."
                )

    def _update_dataloader_indices(self, keys_for_epoch):
        # ADAPTATION: Helper function to update indices robustly, checking attributes relevant to nnU-Net loaders.
        def update_loader(loader):
            if loader is None: return False
            # Check for 'indices' attribute (common in custom DataLoaders)
            if hasattr(loader, 'indices'):
                loader.indices = keys_for_epoch
                return True
            # Check the 'data' attribute which often holds the dataset/indices in batchgenerators style
            if hasattr(loader, 'data') and hasattr(loader.data, 'indices'):
                 loader.data.indices = keys_for_epoch
                 return True
            # Check for 'dataset.indices' (common in standard PyTorch)
            if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'indices'):
                loader.dataset.indices = keys_for_epoch
                return True
            return False

        # Check both potential raw loader names (dl_tr_raw used in user's code, dl_tr as fallback)
        raw_loader = getattr(self, 'dl_tr_raw', getattr(self, 'dl_tr', None))
        updated_dl_tr = update_loader(raw_loader)
        
        # Update the augmenter's internal data_loader
        tr_gen_loader = getattr(getattr(self, 'tr_gen', None), 'data_loader', None)
        updated_tr_gen = update_loader(tr_gen_loader)
        
        # Ensure at least one succeeded
        return updated_dl_tr or updated_tr_gen


    def on_train_epoch_start(self):
        """
        Implements the curriculum logic.
        Uses Global Shuffle for RANDOM/Filtering, and Ordered Interleaving for Curriculum strategies.
        """
        
        # 1. Standard epoch start procedures.
        if hasattr(self, 'network'):
            self.network.train()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.current_epoch)

        # Check if Dynamic Sampling is enabled
        if not getattr(self, 'enable_custom_sampling', False):
            return

        # Robustly ensure keys are available (Key Recovery Logic)
        if not hasattr(self, 'positive_keys_tr') or not hasattr(self, 'negative_keys_tr'):
            if hasattr(self, 'classify_keys'):
                 self.print_to_log_file("INFO: Positive/Negative keys missing. Attempting recovery.")
                 try:
                     # Determine the dataset source robustly
                     dataset_to_use = getattr(self, 'dataset_tr', None)
                     
                     # ADAPTATION: Check dl_tr_raw if dataset_tr is missing (matching parent implementation)
                     if dataset_to_use is None:
                          # Check both raw loader names
                          raw_loader = getattr(self, 'dl_tr_raw', getattr(self, 'dl_tr', None))
                          # Check common attributes where the dataset might be stored
                          if raw_loader:
                               # In the user's implementation, the dataset is passed as 'data' to the loader
                               dataset_to_use = getattr(raw_loader, 'data', getattr(raw_loader, 'dataset', None))


                     if dataset_to_use:
                        self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_to_use)
                     else:
                         raise ValueError("Cannot access training dataset for key classification.")
                         
                 except Exception as e:
                     self.print_to_log_file(f"ERROR: Failed to classify keys during on_train_epoch_start: {e}. Skipping epoch sampling.")
                     return
            else:
                self.print_to_log_file("ERROR: Missing positive/negative keys and cannot re-classify. Skipping epoch sampling.")
                return
        
        # Ensure the ratio exists. We check for the attributes used in the parent DynamicSampling implementation.
        if hasattr(self, 'negative_to_positive_ratio'):
             ratio = self.negative_to_positive_ratio
        # Calculate ratio if the parent implementation uses fixed negative count instead of ratio
        elif hasattr(self, 'num_negative_samples_per_epoch') and hasattr(self, 'positive_keys_tr') and len(self.positive_keys_tr) > 0:
             ratio = self.num_negative_samples_per_epoch / len(self.positive_keys_tr)
        else:
             ratio = 1.0 # Default fallback


        # --- Curriculum Dynamic Sampling Logic ---
        
        positive_keys = list(self.positive_keys_tr)
        negative_keys = list(self.negative_keys_tr)
        
        strategy = self.strategy_name
        metric = self.strategy_metric

        # Force RANDOM if manager failed
        if not self.curriculum_manager or not self.curriculum_manager.is_initialized:
             strategy = "RANDOM"

        self.print_to_log_file(f"Epoch {self.current_epoch}: Applying Strategy: {strategy} (Metric: {metric})")
        
        # 2. Apply Curriculum Strategy (Filtering and Sorting)
        
        # --- A. Baseline/Random ---
        if strategy == "RANDOM":
            # In RANDOM mode, we shuffle the positives here. The combination will use Global Shuffle.
            self.RNG.shuffle(positive_keys)
            processed_positive_keys = positive_keys

        # --- B. Cycling Strategy ---
        elif strategy == "CYCLING":
            if self.curriculum_schedule:
                strategy_index = self.current_epoch % len(self.curriculum_schedule)
                current = self.curriculum_schedule[strategy_index]
                self.print_to_log_file(f"  Cycling Step: {current['metric']} ({current['order']})")
                processed_positive_keys = self.curriculum_manager.get_ordered_keys(
                    current['metric'], positive_keys, order=current['order']
                )
            else: 
                 # Fallback if schedule is empty (e.g., manager failed during CYCLING setup)
                 self.RNG.shuffle(positive_keys)
                 processed_positive_keys = positive_keys

        # --- C. Static Ordering Strategies ---
        elif strategy in ["E2H", "H2E", "MEDIUM_FIRST", "EXTREMES_FIRST"]:
            processed_positive_keys = self.curriculum_manager.get_ordered_keys(metric, positive_keys, order=strategy)

        # --- D. Static Filtering Strategies ---
        elif strategy == "HARDEST_PCT" or strategy == "EASIEST_PCT":
            is_hardest = (strategy == "HARDEST_PCT")
            order = "H2E" if is_hardest else "E2H"
            
            sorted_keys = self.curriculum_manager.get_ordered_keys(metric, positive_keys, order=order)
            
            subset_size = int(len(sorted_keys) * self.strategy_percentage)
            if subset_size == 0 and len(sorted_keys) > 0: subset_size = 1
                
            processed_positive_keys = sorted_keys[:subset_size]
            self.print_to_log_file(f"  Static Filtering: Keeping {subset_size}/{len(positive_keys)} positive cases (Top {self.strategy_percentage*100:.1f}%)")
            
            # We shuffle the subset here. Since the order is randomized, we will use Global Shuffle later.
            self.RNG.shuffle(processed_positive_keys) 

        # --- E. Dynamic Pacing Strategies ---
        elif strategy == "STEPPED_E2H":
            # (Logic remains the same - requires ordered interleaving)
            sorted_keys = self.curriculum_manager.get_ordered_keys(metric, positive_keys, order="E2H")
            num_steps = self.strategy_steps
            total_epochs = getattr(self, 'num_epochs', 1000)
            step_size = max(1, total_epochs / num_steps)
            
            current_step = int(self.current_epoch // step_size)
            current_step = min(current_step, num_steps - 1)
            fraction_to_include = (current_step + 1) / num_steps
            
            num_keys_to_include = int(np.ceil(len(sorted_keys) * fraction_to_include))
            processed_positive_keys = sorted_keys[:num_keys_to_include]
            self.print_to_log_file(f"  Stepped Pacing Step {current_step+1}/{num_steps}. Using {len(processed_positive_keys)}/{len(sorted_keys)} cases.")

        elif strategy == "MEDIUM_OUT":
            # (Logic remains the same - requires ordered interleaving)
            scores = self.curriculum_manager._get_scores_for_keys(metric, positive_keys)
            if scores is None:
                # Fallback to random if scores are unavailable
                self.RNG.shuffle(positive_keys)
                processed_positive_keys = positive_keys
            else:
                percentiles = scores.rank(pct=True)
                W_min = 0.1
                total_epochs = getattr(self, 'num_epochs', 1000)
                progress = self.current_epoch / max(1, (total_epochs - 1))
                W_t = W_min + (1.0 - W_min) * progress
                
                lower_bound = 0.5 - W_t / 2
                upper_bound = 0.5 + W_t / 2
                
                mask = (percentiles >= lower_bound) & (percentiles <= upper_bound)
                selected_keys = percentiles[mask]
                
                # Sort the selected keys (E2H within the window)
                processed_positive_keys = selected_keys.sort_values(ascending=True).index.tolist()
                self.print_to_log_file(f"  Medium-Out Window: [{lower_bound:.2f}, {upper_bound:.2f}] (Size {W_t:.2f}). Selected {len(processed_positive_keys)}/{len(positive_keys)} cases.")

        else:
            # Fallback
            self.RNG.shuffle(positive_keys)
            processed_positive_keys = positive_keys


        # 3. Handle Negative Keys (Standard DS implementation)
        num_pos = len(processed_positive_keys)
        
        # Use the calculated ratio for sampling negatives
        num_neg_to_sample = int(np.round(num_pos * ratio))
        
        # Constrain the number of negatives if the parent DS trainer specifies a limit (as seen in the baseline structure)
        if hasattr(self, 'num_negative_samples_per_epoch'):
             max_neg = self.num_negative_samples_per_epoch
             # Ensure max_neg is treated as an integer
             if not isinstance(max_neg, (int, float)) or isinstance(max_neg, bool):
                 try:
                     max_neg = int(max_neg)
                 except (ValueError, TypeError):
                     max_neg = float('inf') # If conversion fails, assume no limit
                     
             if num_neg_to_sample > max_neg:
                  self.print_to_log_file(f"INFO: Limiting negative samples from {num_neg_to_sample} to the DS limit of {max_neg}.")
                  num_neg_to_sample = int(max_neg)

        
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
        
        # Shuffle the negative keys (important for both Global Shuffle and Interleaving)
        self.RNG.shuffle(sampled_negative_keys)

        # 4. Combine Keys (CRITICAL FIX: Global Shuffle vs. Ordered Interleaving)
        
        # Define strategies that require strict ordering (and thus Ordered Interleaving)
        ORDERING_REQUIRED = [
            "CYCLING", "E2H", "H2E", "MEDIUM_FIRST", "EXTREMES_FIRST", 
            "STEPPED_E2H", "MEDIUM_OUT"
        ]

        # Determine if interleaving is needed. Handle specific fallbacks.
        use_interleaving = False
        if strategy in ORDERING_REQUIRED:
            use_interleaving = True
            # Check for specific fallbacks where ordering was lost
            if strategy == "CYCLING" and not self.curriculum_schedule:
                 use_interleaving = False
            # Check if MEDIUM_OUT fell back to random shuffle due to missing scores
            elif strategy == "MEDIUM_OUT" and 'processed_positive_keys' in locals() and processed_positive_keys == positive_keys:
                 # If the lists are identical, it means the fallback occurred
                 use_interleaving = False
        
        keys_for_epoch = []
        
        if use_interleaving:
            # --- Ordered Interleaving (Maintains curriculum order) ---
            neg_ptr = 0
            # Recalculate ratio based on actual sampled negatives for accurate interleaving
            if num_pos > 0:
                interleave_ratio = len(sampled_negative_keys) / num_pos
            else:
                interleave_ratio = 0

            if num_pos == 0:
                 keys_for_epoch = sampled_negative_keys
            else:
                for i, pos_key in enumerate(processed_positive_keys):
                    keys_for_epoch.append(pos_key)
                    ideal_neg_count = round((i + 1) * interleave_ratio)
                    neg_to_insert = ideal_neg_count - neg_ptr
                    
                    for j in range(neg_to_insert):
                        if neg_ptr < len(sampled_negative_keys):
                            keys_for_epoch.append(sampled_negative_keys[neg_ptr])
                            neg_ptr += 1

                # Append any remaining negatives
                while neg_ptr < len(sampled_negative_keys):
                    keys_for_epoch.append(sampled_negative_keys[neg_ptr])
                    neg_ptr += 1
            
            combination_method = "Ordered Interleaving"

        else:
            # --- Global Shuffle (For RANDOM, HARDEST_PCT, EASIEST_PCT, or Fallbacks) ---
            # This matches standard Dynamic Sampling behavior exactly.
            keys_for_epoch = processed_positive_keys + sampled_negative_keys
            self.RNG.shuffle(keys_for_epoch)
            combination_method = "Global Shuffle"


        # 5. Update the data loader
        self.print_to_log_file(f"Epoch {self.current_epoch} sampling summary: {len(keys_for_epoch)} cases ({num_pos} pos, {len(sampled_negative_keys)} neg). Method: {combination_method}.")

        if not self._update_dataloader_indices(keys_for_epoch):
             self.print_to_log_file("CRITICAL ERROR: Could not update data loader indices.")
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
