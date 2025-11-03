# Filename: nnUNetTrainer_DS_TO_Curriculum.py
import torch
import numpy as np
import pandas as pd  # Added for curriculum loading
import os            # Added for path joining

# --- FIX: Update imports based on user-specified filenames and use aliases for clarity ---

try:
    from .nnUNetTrainer_DynamicSampling_targeted import nnUNetTrainer_DynamicSampling_targeted as nnUNetTrainer_DynamicSampling
except ImportError:
     print("ERROR: Cannot import nnUNetTrainer_DynamicSampling from nnUNetTrainer_DynamicSampling_targeted.py. Check filename, location, and class name."); raise

try:
    from .nnUNetDataLoader_Targeted_dynamic import nnUNetDataLoader_Targeted
except ImportError:
    print("ERROR: Cannot import nnUNetDataLoader_Targeted from nnUNetDataLoader_Targeted_dynamic.py. Check filename, location, and class name."); raise

# Import standard components required for get_dataloaders
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader # Needed for validation loader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.utilities.label_handling.label_handling import LabelManager


class nnUNetTrainer_DS_TO_Curriculum(nnUNetTrainer_DynamicSampling):
    """
    Combines:
    1. Dynamic Sampling (DS): Case-level sampling to balance positive/negative cases.
    2. Targeted Oversampling (TO): Patch-level oversampling of a specific target class.
    3. Curriculum Learning (CL): Case-level pacing for positive cases, from "easy" to "hard"
       and back, based on quantitative metrics from an external CSV file.
    
    Designed for high-impact research (e.g., CVPR).
    """
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        
        super().__init__(plans, configuration, fold, dataset_json, device)

        # --- Targeted Oversampling (TO) Configuration ---
        self.target_sampling_ratio = 0.75 
        self.oversample_foreground_percent = 0.66
        
        # --- NEW: Curriculum Learning (CL) Configuration ---
        self.enable_curriculum_learning = True
        
        # Define the path to the CSV file containing difficulty metrics.
        # Assumes the CSV is placed inside the preprocessed dataset folder.
        self.curriculum_csv_path = os.path.join(self.preprocessed_dataset_folder, "PanTS_reports.csv")
        
        # Select your curriculum strategy from the 10+ defined in `calculate_difficulty_scores`.
        # This is a key hyperparameter for your experiments.
        self.curriculum_strategy = "CompositeDifficultyScore"  # <-- CHANGE THIS TO EXPERIMENT
        
        # Number of epochs for ONE phase (e.g., easy-to-hard).
        # A full cycle (easy-hard-easy) will take 2 * curriculum_epoch_cycle epochs.
        self.curriculum_epoch_cycle = 50 
        
        # This will store the sorted list of positive case keys (e.g., ['PanTS_0001', 'PanTS_0005', ...])
        self.positive_keys_sorted = []
        # ---------------------------------------------------

        self.print_to_log_file("Initialized Hybrid nnUNetTrainer_DS_TO_Curriculum.")
        self.print_to_log_file(f"  TO Config: Ratio={self.target_sampling_ratio}, Fg Pct={self.oversample_foreground_percent}.")
        if self.enable_curriculum_learning:
            self.print_to_log_file(f"  CL Config: Enabled, Strategy='{self.curriculum_strategy}', Cycle={self.curriculum_epoch_cycle} epochs.")
            self.print_to_log_file(f"  CL CSV Path: {self.curriculum_csv_path}")

    
    def get_dataloaders(self):
        """
        Overrides the base method to:
        1. Inject the Targeted DataLoader (nnUNetDataLoader_Targeted) for training.
        2. Keep the standard DataLoader for validation.
        3. Perform Dynamic Sampling classification (positive/negative keys).
        4. NEW: Initialize the Curriculum Learning module by loading, scoring, and sorting positive cases.
        """
        
        # 1. Perform standard configuration setup
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

        # 2. Instantiate the Targeted DataLoader for Training (TO Injection)
        dl_tr = nnUNetDataLoader_Targeted(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            target_label_id=self.target_label_id, 
            target_sampling_ratio=self.target_sampling_ratio
        )
        
        # 3. Instantiate the Standard DataLoader for Validation
        dl_val = nnUNetDataLoader(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                  self.configuration_manager.patch_size, self.label_manager,
                                  oversample_foreground_percent=self.oversample_foreground_percent,
                                  transforms=val_transforms, 
                                  probabilistic_oversampling=self.probabilistic_oversampling)

        # 4. Store the raw loaders
        self.dl_tr_raw = dl_tr
        self.dl_val_raw = dl_val

        # 5. Perform Dynamic Sampling Classification (DS Logic)
        if self.enable_custom_sampling:
            if hasattr(dl_tr, 'annotated_classes_key'):
                self.annotated_classes_key_tr = dl_tr.annotated_classes_key

            self.positive_keys_tr, self.negative_keys_tr = self.classify_keys(dataset_tr)
            self.print_to_log_file(f"Training set analysis: {len(self.positive_keys_tr)} positive, {len(self.negative_keys_tr)} negative.")

            if hasattr(self, 'num_negative_samples_per_epoch') and len(self.negative_keys_tr) < self.num_negative_samples_per_epoch:
                 self.print_to_log_file(f"INFO: Total negative samples ({len(self.negative_keys_tr)}) is less than the requested subset ({self.num_negative_samples_per_epoch}). Using all.")

        # 6. NEW: Initialize Curriculum Learning
        if self.enable_curriculum_learning and self.enable_custom_sampling:
            self.initialize_curriculum()

        # Return (None, None) as required by the Dynamic Sampling lifecycle management
        return None, None

    # ---------------------------------------------------------------
    # --- NEW: CURRICULUM LEARNING (CL) METHODS ---------------------
    # ---------------------------------------------------------------

    def initialize_curriculum(self):
        """
        Loads the CSV, calculates difficulty scores, and creates the sorted list
        of positive keys ('self.positive_keys_sorted').
        This is called once from `get_dataloaders`.
        """
        self.print_to_log_file("Initializing Curriculum Learning...")
        
        try:
            df = pd.read_csv(self.curriculum_csv_path)
        except FileNotFoundError:
            self.print_to_log_file(f"CRITICAL ERROR: Curriculum CSV not found at {self.curriculum_csv_path}")
            self.print_to_log_file("Curriculum Learning DISABLED.")
            self.enable_curriculum_learning = False
            return
        except Exception as e:
            self.print_to_log_file(f"CRITICAL ERROR: Failed to load curriculum CSV: {e}")
            self.print_to_log_file("Curriculum Learning DISABLED.")
            self.enable_curriculum_learning = False
            return

        # Calculate scores for all cases in the CSV
        try:
            difficulty_scores = self.calculate_difficulty_scores(df)
        except Exception as e:
            self.print_to_log_file(f"CRITICAL ERROR: Failed to calculate difficulty scores (Strategy: {self.curriculum_strategy}): {e}")
            self.print_to_log_file("Check CSV headers and strategy name.")
            self.print_to_log_file("Curriculum Learning DISABLED.")
            self.enable_curriculum_learning = False
            return

        # Create a list of tuples (key, score) for *our* positive cases
        scored_positive_keys = []
        missing_keys = 0
        for key in self.positive_keys_tr:
            # We must match the key format (e.g., 'PanTS_0001') with the CSV 'PanTS ID'
            if key not in difficulty_scores:
                missing_keys += 1
            else:
                scored_positive_keys.append((key, difficulty_scores[key]))
        
        if missing_keys > 0:
            self.print_to_log_file(f"WARNING: {missing_keys} positive keys were not found in the curriculum CSV. They will be excluded from CL.")

        if not scored_positive_keys:
            self.print_to_log_file("CRITICAL ERROR: No positive keys matched the curriculum CSV. CL disabled.")
            self.enable_curriculum_learning = False
            return
            
        # Sort the list by score (ascending: easy -> hard)
        # Note: 'calculate_difficulty_scores' defines "easy" as a *low* score.
        scored_positive_keys.sort(key=lambda x: x[1])
        
        # Store just the sorted keys
        self.positive_keys_sorted = [key for key, score in scored_positive_keys]
        
        self.print_to_log_file(f"Curriculum Initialized. {len(self.positive_keys_sorted)} positive cases scored and sorted.")
        self.print_to_log_file(f"  Easiest case: {self.positive_keys_sorted[0]} (Score: {scored_positive_keys[0][1]:.4f})")
        self.print_to_log_file(f"  Hardest case: {self.positive_keys_sorted[-1]} (Score: {scored_positive_keys[-1][1]:.4f})")

    
    def calculate_difficulty_scores(self, df: pd.DataFrame) -> dict:
        """
        *** CREATIVE CORE FOR CVPR ***
        
        Implements 10+ curriculum strategies.
        Each strategy returns a dictionary: {'case_id': difficulty_score}
        
        Convention: LOW score = EASY case, HIGH score = HARD case.
        
        Args:
            df (pd.DataFrame): The loaded CSV (e.g., PanTS_reports.csv)
        
        Returns:
            dict: A dictionary mapping 'PanTS ID' to a numeric difficulty score.
        """
        self.print_to_log_file(f"Calculating difficulty scores using strategy: {self.curriculum_strategy}")
        
        # Helper: Normalize a series (z-score)
        def z_score(series: pd.Series) -> pd.Series:
            return (series - series.mean()) / (series.std() + 1e-8)

        # Helper: Inverse a series (so 1/large_volume = small_score)
        def inverse_score(series: pd.Series) -> pd.Series:
            return 1 / (series + 1e-8)

        # Set index for easy mapping
        # We assume the keys in nnU-Net (self.positive_keys_tr) match 'PanTS ID'
        if 'PanTS ID' not in df.columns:
            raise KeyError("CSV must contain a 'PanTS ID' column.")
        df = df.set_index('PanTS ID')

        # --- Define Strategies ---
        
        if self.curriculum_strategy == 'LargestTumorSize':
            # Easy = Large diameter. Hard = Small diameter.
            # Score = 1 / diameter (so small diameter = high score = hard)
            metric = df['largest pancreatic PDAC diameter (cm)'].fillna(0)
            scores = inverse_score(metric)
        
        elif self.curriculum_strategy == 'TotalTumorVolume':
            # Easy = Large volume. Hard = Small volume.
            # Score = 1 / volume
            metric = df['total pancreatic PDAC volume (cm^3)'].fillna(0)
            scores = inverse_score(metric)

        elif self.curriculum_strategy == 'TumorInstanceCount':
            # Easy = 1 instance. Hard = Many instances.
            # Score = count
            metric = df['number of pancreatic PDAC instances'].fillna(0)
            scores = metric
            
        elif self.curriculum_strategy == 'RelativeTumorBurden':
            # Easy = High tumor/organ ratio. Hard = Low ratio.
            # Score = 1 / ratio
            vol = df['total pancreatic PDAC volume (cm^3)'].fillna(0)
            organ_vol = df['pancreas volume (cm^3)'].fillna(1) # Avoid div by zero
            ratio = vol / (organ_vol + 1e-8)
            scores = inverse_score(ratio)

        elif self.curriculum_strategy == 'SmallestTumorFirst':
            # "Anti-Curriculum" or "Hard-Mining"
            # Easy = Small diameter. Hard = Large diameter.
            # Score = diameter
            metric = df['largest pancreatic PDAC diameter (cm)'].fillna(0)
            scores = metric

        elif self.curriculum_strategy == 'PatientAge':
            # Hypothesis: Older patients are "harder" (e.g., more comorbidities)
            # Easy = Young. Hard = Old.
            # Score = age
            metric = df['age'].fillna(df['age'].median())
            scores = metric
            
        elif self.curriculum_strategy == 'ScannerFrequency':
            # "Domain Curriculum"
            # Easy = Common scanner. Hard = Rare scanner.
            # Score = 1 / frequency
            counts = df['scanner'].value_counts(normalize=True)
            metric = df['scanner'].map(counts)
            scores = inverse_score(metric.fillna(1.0)) # Fill unseen with max freq

        elif self.curriculum_strategy == 'DistractorCount':
            # Hypothesis: More *non-target* lesions make finding the target harder.
            # Easy = Few distractors. Hard = Many distractors.
            # Score = number of *lesions* + number of *cysts*
            lesions = df['number of pancreatic lesion instances'].fillna(0)
            cysts = df['number of pancreatic cyst instances'].fillna(0)
            pdac = df['number of pancreatic PDAC instances'].fillna(0)
            # Distractors = Total Lesions - Target Lesions
            distractor_count = (lesions + cysts) - pdac
            scores = distractor_count.clip(lower=0) # Ensure non-negative

        elif self.curriculum_strategy == 'TumorLocation':
            # Hypothesis: Location impacts difficulty (e.g., tail is harder).
            # Easy = Head. Medium = Body. Hard = Tail.
            # Score = 1 (head), 2 (body), 3 (tail)
            location_map = {'head': 1, 'body': 2, 'tail': 3}
            metric = df['largest pancreatic PDAC location (head, body, tail)'].map(location_map)
            scores = metric.fillna(2) # Default to 'body' if missing
            
        elif self.curriculum_strategy == 'CompositeDifficultyScore':
            # ** ADVANCED STRATEGY **
            # Combines multiple metrics.
            # Hard = Small Volume + Many Instances + Low Relative Burden
            
            # 1. Inverse Volume (Hard = High Score)
            s_vol = z_score(inverse_score(df['total pancreatic PDAC volume (cm^3)'].fillna(0)))
            
            # 2. Instance Count (Hard = High Score)
            s_count = z_score(df['number of pancreatic PDAC instances'].fillna(0))
            
            # 3. Inverse Relative Burden (Hard = High Score)
            vol = df['total pancreatic PDAC volume (cm^3)'].fillna(0)
            organ_vol = df['pancreas volume (cm^3)'].fillna(1)
            ratio = vol / (organ_vol + 1e-8)
            s_burden = z_score(inverse_score(ratio))
            
            # Weighted average. Weights are hyperparameters to tune.
            scores = (0.4 * s_vol) + (0.3 * s_count) + (0.3 * s_burden)

        elif self.curriculum_strategy == 'LiverMetastasisComposite':
            # Example for a *different* problem (e.g., finding liver mets)
            # Hard = Small liver tumors + Many instances + Large Liver
            s_vol = z_score(inverse_score(df['total liver tumor volume (cm^3)'].fillna(0)))
            s_count = z_score(df['number of liver tumor instances'].fillna(0))
            s_organ = z_score(df['liver volume (cm^3)'].fillna(df['liver volume (cm^3)'].median()))
            scores = (0.4 * s_vol) + (0.4 * s_count) + (0.2 * s_organ)
            
        elif self.curriculum_strategy == 'Random':
            # Control experiment: No curriculum, just a random (but stable) order.
            scores = pd.Series(np.random.rand(len(df)), index=df.index)

        else:
            raise ValueError(f"Unknown curriculum_strategy: {self.curriculum_strategy}")

        # Return as a dictionary
        return scores.to_dict()


    def on_train_epoch_start(self):
        """
        Overrides the base class method to inject curriculum-based key selection.
        Called at the beginning of each training epoch.
        """
        if not self.enable_custom_sampling:
            # If DS is off, CL is also off. Defer to parent.
            super().on_train_epoch_start()
            return

        # 1. Handle Negative Keys (Standard DS Logic)
        # Select a random subset of negative keys for this epoch
        np.random.shuffle(self.negative_keys_tr)
        selected_negative_keys = self.negative_keys_tr[:self.num_negative_samples_per_epoch]

        # 2. Handle Positive Keys (NEW CL Logic)
        if not self.enable_curriculum_learning or not self.positive_keys_sorted:
            # Fallback: CL is disabled or failed, use standard random sampling
            np.random.shuffle(self.positive_keys_tr)
            selected_positive_keys = self.positive_keys_tr
            log_msg = "CL disabled/failed. Using all positive keys."
        else:
            # Get the paced subset of positive keys for this epoch
            selected_positive_keys = self.get_curriculum_positive_keys()
            log_msg = f"CL Strategy '{self.curriculum_strategy}'. Selecting {len(selected_positive_keys)}/{len(self.positive_keys_sorted)} positive keys."

        # 3. Combine and Update Data Loader
        selected_keys = selected_positive_keys + selected_negative_keys
        np.random.shuffle(selected_keys) # Shuffle the final list
        
        # This 'update_keys' method must exist in your nnUNetDataLoader_Targeted
        try:
            self.dl_tr_raw.update_keys(selected_keys)
        except AttributeError:
            self.print_to_log_file("CRITICAL ERROR: self.dl_tr_raw (nnUNetDataLoader_Targeted) must have an 'update_keys(keys_list)' method.")
            raise

        # Log the epoch's sampling
        self.print_to_log_file(f"Epoch {self.current_epoch}: {log_msg}")
        self.print_to_log_file(f"  Selected {len(selected_positive_keys)} positive, {len(selected_negative_keys)} negative. Total: {len(selected_keys)}")


    def get_curriculum_positive_keys(self) -> list:
        """
        Implements the "easy-to-hard-to-easy" pacing.
        Calculates the subset of `self.positive_keys_sorted` to use for the current epoch.
        """
        
        total_epochs_in_cycle = self.curriculum_epoch_cycle * 2
        # What epoch are we in, *within the current cycle*?
        current_cycle_epoch = self.current_epoch % total_epochs_in_cycle
        
        num_positive_total = len(self.positive_keys_sorted)
        
        # Define a minimum pool of cases (e.g., 10%) to always be present
        # This aids stability, especially in the "hard-to-easy" phase.
        min_cases = max(1, int(num_positive_total * 0.1))

        if current_cycle_epoch < self.curriculum_epoch_cycle:
            # --- Phase 1: Easy-to-Hard (Growing pool) ---
            # Progress goes from (e.g.) 1/50 to 50/50
            progress = (current_cycle_epoch + 1) / self.curriculum_epoch_cycle
            num_to_take = int(np.ceil(num_positive_total * progress))
            
            # Log phase
            if current_cycle_epoch == 0:
                self.print_to_log_file("  Curriculum Phase: Starting Easy-to-Hard.")

        else:
            # --- Phase 2: Hard-to-Easy (Shrinking pool) ---
            # Progress goes from (e.g.) 0/50 to 49/50
            progress = (current_cycle_epoch - self.curriculum_epoch_cycle) / self.curriculum_epoch_cycle
            # Invert progress to go from 1.0 down to ~0.0
            inverted_progress = 1.0 - progress
            num_to_take = int(np.ceil(num_positive_total * inverted_progress))

            # Log phase
            if current_cycle_epoch == self.curriculum_epoch_cycle:
                self.print_to_log_file("  Curriculum Phase: Starting Hard-to-Easy.")
        
        # Ensure we always take at least the minimum number of cases
        num_to_take = max(min_cases, num_to_take)
        
        # Return the slice of the sorted list (from easiest up to hardest)
        return self.positive_keys_sorted[:num_to_take]

