# Filename: nnUNetTrainer_Ablation_Curriculum.py
import torch
import numpy as np
import pandas as pd
import os
import re
import inspect
from typing import List, Dict, Any, Optional

# Import the base trainer to patch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# --------------------------------------------------------------------------------------------------
# DYNAMIC COMPATIBILITY PATCH FOR RIGID BASE TRAINER
# --------------------------------------------------------------------------------------------------
# This patch addresses environments where nnUNetTrainer has an outdated signature.

# Inspect the signature to determine if patching is needed.
_original_signature = inspect.signature(nnUNetTrainer.__init__)
_params = list(_original_signature.parameters.keys())
# Check if 'kwargs' is missing AND if modern arguments like 'unpack_dataset' are missing.
needs_patch = ('kwargs' not in _params) and ('unpack_dataset' not in _params or 'device' not in _params)

if needs_patch:
    print("INFO: Base nnUNetTrainer signature requires compatibility patch. Applying dynamically.")
    
    # Store the original init
    _original_nnUNetTrainer_init = nnUNetTrainer.__init__

    # Define the patched init
    def patched_init(self, plans, configuration, fold, dataset_json, *args, **kwargs):
        
        # Reconstruct the arguments for the original call.
        call_args = []
        # Expected positional arguments (excluding 'self')
        expected_pos_args = [p for p in _params if p != 'self']
        # All arguments currently provided positionally
        current_args = [plans, configuration, fold, dataset_json] + list(args)

        # Map current arguments to the expected positional signature.
        for i, arg_name in enumerate(expected_pos_args):
            if i < len(current_args):
                # Use the provided positional argument
                call_args.append(current_args[i])
            elif arg_name in kwargs:
                # Handle case where expected positional arg was passed as a keyword
                call_args.append(kwargs.pop(arg_name))
            # If argument is missing and has a default, it will be handled by the original call.
        
        # Call the original init using the reconstructed positional arguments
        # NOTE: Introspection happens within this call. By renaming **kwargs to **other_args 
        # in the derived class, we avoid the KeyError.
        try:
            _original_nnUNetTrainer_init(self, *call_args)
        except TypeError as e:
            print(f"FATAL: Patch failed during original init call. Args provided: {call_args}. Expected params: {_params}. Error: {e}")
            raise e
        
        # Manually handle arguments ignored by the original init
        
        # Handle unpack_dataset
        if 'unpack_dataset' not in _params:
            unpack_dataset = kwargs.get('unpack_dataset', True)
            if unpack_dataset:
                    if not hasattr(self, 'unpack_dataset') or self.unpack_dataset is False:
                        self.unpack_dataset = True

        # Handle device
        if 'device' not in _params:
            device = kwargs.get('device', torch.device('cuda'))
            if not hasattr(self, 'device') or self.device is None:
                 self.device = device

    # Apply the patch
    nnUNetTrainer.__init__ = patched_init

# --------------------------------------------------------------------------------------------------
# Import Foundation Trainer (Post-Patch)
# --------------------------------------------------------------------------------------------------

# Import the foundational DS_TO trainer
try:
    from .nnUNetTrainer_DS_TO import nnUNetTrainer_DS_TO as nnUNetTrainer_DS_TO_Foundation
except ImportError:
    print("FATAL: Could not import nnUNetTrainer_DS_TO.")
    # Fallback definition using the (now patched) base trainer
    nnUNetTrainer_DS_TO_Foundation = nnUNetTrainer

# --------------------------------------------------------------------------------------------------
# Curriculum Manager Utility (Enhanced for Ablation Studies)
# --------------------------------------------------------------------------------------------------

class CurriculumManagerAblation:
    """Handles metadata loading, difficulty scoring, and advanced key ordering/filtering."""
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
        if valid_scores.empty:
             scores['S10_Composite'] = 0
        else:
            ranks = valid_scores.rank(method='average', ascending=True)
            scores['S10_Composite'] = ranks.sum(axis=1)

        self.scores = scores
        self.strategy_metrics = scores.columns.tolist()
        return True

    # Helper functions for ordering
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
# Curriculum Trainer (Ablation Implementation - FINALIZED)
# --------------------------------------------------------------------------------------------------

class nnUNetTrainer_Ablation_Curriculum(nnUNetTrainer_DS_TO_Foundation):

    # FIX: Rename **kwargs to **other_args to hide it from the base trainer's introspection.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **other_args):
        
        # Infer the preprocessed folder path early for CSV finding.
        if 'nnUNet_preprocessed' in os.environ:
            dataset_name = dataset_json.get('name', f"Dataset{str(dataset_json.get('dataset_id', 'Unknown'))}")
            self.preprocessed_folder_temp = os.path.join(os.environ['nnUNet_preprocessed'], dataset_name)
        else:
            self.preprocessed_folder_temp = None
            print("WARNING: nnUNet_preprocessed environment variable not set.")


        # --- Curriculum Setup (Must happen before super().__init__) ---
        
        # 1. Initialize Curriculum Manager
        self._setup_curriculum_manager()

        # 2. Configure Strategy
        self._configure_ablation_strategy()

        # 3. Standard Initialization
        # FIX: Pass the renamed arguments up the chain.
        super().__init__(plans, configuration, fold, dataset_json, **other_args)

        # 4. Post-Initialization Logging and RNG setup
        
        # Logging
        log_func = getattr(self, 'print_to_log_file', print)
        if self.curriculum_manager and self.curriculum_manager.is_initialized:
            log_func(f"Curriculum Learning initialized. Strategy: {self.strategy_name}, Metric: {self.strategy_metric}")
        elif self.strategy_name != "RANDOM":
            log_func("WARNING: CurriculumManager failed. Strategy forced to RANDOM.")
            self.strategy_name = "RANDOM"

        # Ensure RNG uses the modern numpy generator (required for curriculum logic)
        if hasattr(self, 'RNG') and self.RNG is not None:
            if not isinstance(self.RNG, np.random.Generator):
                print("INFO: Converting legacy RNG (RandomState) to modern numpy Generator.")
                try:
                    # Attempt robust seed extraction
                    state = self.RNG.get_state()
                    if isinstance(state, tuple) and len(state) > 1 and isinstance(state[1], np.ndarray) and state[1].size > 0:
                         seed = state[1][0]
                    else:
                         raise ValueError("Cannot extract seed.")
                except Exception:
                    fold_val = getattr(self, 'fold', 0)
                    seed = fold_val if isinstance(fold_val, int) else 0
                self.RNG = np.random.default_rng(seed)
        else:
            # Defensive initialization
            fold_val = getattr(self, 'fold', 0)
            seed = fold_val if isinstance(fold_val, int) else 0
            self.RNG = np.random.default_rng(seed)

    # (The rest of the methods remain the same as the previous robust versions)

    def _setup_curriculum_manager(self):
        # Path finding and manager initialization logic
        csv_filename = "PanTS_reports.csv"
        csv_path = None
        search_locations = []

        if self.preprocessed_folder_temp:
            search_locations.append(self.preprocessed_folder_temp)
            # Check up to 2 parent directories
            parent_dir = os.path.dirname(self.preprocessed_folder_temp)
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


    def _configure_ablation_strategy(self):
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
        # ADAPTATION: Handle the initialization sequence specific to the DS_TO lifecycle.
        log_func = getattr(self, 'print_to_log_file', print)

        # --- 1. Bridging ---
        # Populate standard attributes from custom attributes.
        
        if getattr(self, 'dataloader_train', None) is None:
            if hasattr(self, 'dl_tr_raw') and self.dl_tr_raw is not None:
                 log_func("INFO: Bridging DS/TO initialization: Populating self.dataloader_train from self.dl_tr_raw.")
                 self.dataloader_train = self.dl_tr_raw

        if getattr(self, 'dataloader_val', None) is None:
            if hasattr(self, 'dl_val_raw') and self.dl_val_raw is not None:
                 self.dataloader_val = self.dl_val_raw

        # --- 2. Call Parent Implementation ---
        # This sets up augmentation (tr_gen, val_gen).
        super().on_train_start()

        # --- 3. Re-linking ---
        # Ensure that the attributes point to the augmenters (tr_gen/val_gen).
        if hasattr(self, 'tr_gen') and self.tr_gen is not None:
            if self.dataloader_train != self.tr_gen:
                log_func("INFO: Re-linking self.dataloader_train to self.tr_gen (Augmenter).")
                self.dataloader_train = self.tr_gen
            
        if hasattr(self, 'val_gen') and self.val_gen is not None:
             if self.dataloader_val != self.val_gen:
                 self.dataloader_val = self.val_gen

        # --- 4. Final Safety Check ---
        if self.dataloader_train is None:
            raise RuntimeError("Training initialization failed. self.dataloader_train is None after on_train_start().")

    # Use the robust update helper defined in the parent (DynamicSampling_targeted)
    def _update_dataloader_indices(self, keys_for_epoch):
         # This relies on the _update_dataloader_indices_robust method implemented in the parent chain.
         return self._update_dataloader_indices_robust(keys_for_epoch)


    def on_train_epoch_start(self):
        """
        Implements the curriculum logic with Fast Path Optimization for RANDOM.
        """
        log_func = getattr(self, 'print_to_log_file', print)

        # --- Fast Path for RANDOM Strategy (Optimization) ---
        if self.strategy_name == "RANDOM":
            log_func(f"Epoch {self.current_epoch}: Applying Strategy: RANDOM (Fast Path - Parent Implementation)")
            
            # Call the parent's implementation (which handles Global Shuffle)
            super().on_train_epoch_start()
            return

        # --------------------------------------------------------------------
        # --- Full Curriculum Logic (Executed only for non-RANDOM strategies) ---
        # --------------------------------------------------------------------
        
        # 1. Standard epoch start procedures (Manually executed as we override parent)
        if hasattr(self, 'network') and callable(getattr(self.network, 'train', None)):
            self.network.train()

        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step(self.current_epoch)
            except TypeError:
                pass

        # Check prerequisites
        if not getattr(self, 'enable_custom_sampling', False):
            return
        if not hasattr(self, 'positive_keys_tr') or not hasattr(self, 'negative_keys_tr'):
             log_func("ERROR: Missing positive/negative keys.")
             return

        # Determine the sampling ratio and limits
        ratio = getattr(self, 'negative_to_positive_ratio', 1.0)
        
        max_neg_ds = float('inf')
        if hasattr(self, 'num_negative_samples_per_epoch'):
             max_neg_attr = self.num_negative_samples_per_epoch
             if isinstance(max_neg_attr, (int, float)) and not isinstance(max_neg_attr, bool):
                 max_neg_ds = float(max_neg_attr)

        # --- Curriculum Dynamic Sampling Logic ---
        
        positive_keys = list(self.positive_keys_tr)
        negative_keys = list(self.negative_keys_tr)
        
        strategy = self.strategy_name
        metric = self.strategy_metric

        # Check manager status
        if not self.curriculum_manager or not self.curriculum_manager.is_initialized:
             log_func(f"Epoch {self.current_epoch}: Manager failed. Falling back to RANDOM logic.")
             self.RNG.shuffle(positive_keys)
             processed_positive_keys = positive_keys
             strategy = "RANDOM_FALLBACK"
        else:
            log_func(f"Epoch {self.current_epoch}: Applying Strategy: {strategy} (Metric: {metric})")
        
            # 2. Apply Curriculum Strategy (Filtering and Sorting)

            # --- B. Cycling Strategy ---
            if strategy == "CYCLING":
                if self.curriculum_schedule:
                    strategy_index = self.current_epoch % len(self.curriculum_schedule)
                    current = self.curriculum_schedule[strategy_index]
                    log_func(f"  Cycling Step: {current['metric']} ({current['order']})")
                    processed_positive_keys = self.curriculum_manager.get_ordered_keys(
                        current['metric'], positive_keys, order=current['order']
                    )
                else: 
                    # Fallback if schedule is empty
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
                log_func(f"  Static Filtering: Keeping {subset_size}/{len(positive_keys)} positive cases (Top {self.strategy_percentage*100:.1f}%)")
                
                # Shuffle the subset
                self.RNG.shuffle(processed_positive_keys) 

            # --- E. Dynamic Pacing Strategies ---
            elif strategy == "STEPPED_E2H":
                sorted_keys = self.curriculum_manager.get_ordered_keys(metric, positive_keys, order="E2H")
                num_steps = self.strategy_steps
                total_epochs = getattr(self, 'num_epochs', 1000)
                step_size = max(1, total_epochs / num_steps)
                
                current_step = int(self.current_epoch // step_size)
                current_step = min(current_step, num_steps - 1)
                fraction_to_include = (current_step + 1) / num_steps
                
                num_keys_to_include = int(np.ceil(len(sorted_keys) * fraction_to_include))
                processed_positive_keys = sorted_keys[:num_keys_to_include]
                log_func(f"  Stepped Pacing Step {current_step+1}/{num_steps}. Using {len(processed_positive_keys)}/{len(sorted_keys)} cases.")

            elif strategy == "MEDIUM_OUT":
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
                    log_func(f"  Medium-Out Window: [{lower_bound:.2f}, {upper_bound:.2f}] (Size {W_t:.2f}). Selected {len(processed_positive_keys)}/{len(positive_keys)} cases.")


        # 3. Handle Negative Keys
        num_pos = len(processed_positive_keys)
        num_neg_to_sample = int(np.round(num_pos * ratio))
        
        # Constrain the number of negatives
        if num_neg_to_sample > max_neg_ds:
             num_neg_to_sample = int(max_neg_ds)

        if num_neg_to_sample == 0 or len(negative_keys) == 0:
             sampled_negative_keys = []
        elif num_neg_to_sample > len(negative_keys):
             # Oversample negatives
            sampled_negative_keys = negative_keys * (num_neg_to_sample // len(negative_keys))
            remaining = num_neg_to_sample % len(negative_keys)
            if remaining > 0:
                sampled_negative_keys += list(self.RNG.choice(negative_keys, remaining, replace=False))
        else:
            # Undersample negatives
            sampled_negative_keys = list(self.RNG.choice(negative_keys, num_neg_to_sample, replace=False))
        
        self.RNG.shuffle(sampled_negative_keys)

        # 4. Combine Keys (Global Shuffle vs. Ordered Interleaving)
        
        ORDERING_REQUIRED = [
            "CYCLING", "E2H", "H2E", "MEDIUM_FIRST", "EXTREMES_FIRST", 
            "STEPPED_E2H", "MEDIUM_OUT"
        ]

        # Determine if interleaving is needed, handling fallbacks.
        use_interleaving = False
        if strategy in ORDERING_REQUIRED:
            use_interleaving = True
            if strategy == "CYCLING" and not self.curriculum_schedule:
                 use_interleaving = False
            # Check if strategy fell back to random shuffle
            elif 'processed_positive_keys' in locals() and processed_positive_keys == positive_keys:
                 use_interleaving = False
        
        keys_for_epoch = []
        
        if use_interleaving:
            # --- Ordered Interleaving (Maintains curriculum order) ---
            neg_ptr = 0
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
            # --- Global Shuffle (For RANDOM, Filtering, or Fallbacks) ---
            keys_for_epoch = processed_positive_keys + sampled_negative_keys
            self.RNG.shuffle(keys_for_epoch)
            combination_method = "Global Shuffle"


        # 5. Update the data loader
        log_func(f"Epoch {self.current_epoch} sampling summary: {len(keys_for_epoch)} cases ({num_pos} pos, {len(sampled_negative_keys)} neg). Method: {combination_method}.")

        if not self._update_dataloader_indices(keys_for_epoch):
             raise RuntimeError("Failed to configure DataLoader indices for the epoch.")

        # 6. Handle DDP synchronization
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
