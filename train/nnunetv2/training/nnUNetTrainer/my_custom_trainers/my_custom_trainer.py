import torch
import numpy as np

# Import base classes and utilities
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from batchgenerators.utilities.file_and_folder_operations import join, load_pickle

# =================================================================================
# ==> STEP 1: Create a Custom DataLoader with our sampling logic
# =================================================================================
class CustomnnUNetDataLoader(nnUNetDataLoader):
    def __init__(self, data, batch_size, patch_size, final_patch_size, label_manager,
                 oversample_foreground_percent, sampling_probabilities, pad_sides, transforms,
                 positive_indices, negative_indices):
        super().__init__(data, batch_size, patch_size, final_patch_size, label_manager,
                         oversample_foreground_percent, sampling_probabilities, pad_sides, transforms)
        
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.num_neg_to_sample = 1000

        # Handle cases where there are fewer negative samples than requested
        if len(self.negative_indices) < self.num_neg_to_sample:
            print(f"Warning: Requested {self.num_neg_to_sample} negative samples, "
                  f"but only {len(self.negative_indices)} are available. Using all negative samples.")
            self.num_neg_to_sample = len(self.negative_indices)

    def get_indices(self):
        """
        This is the key method we override to implement our custom sampling.
        It's called by the dataloader in each iteration to get a batch of case indices.
        """
        # If the number of indices in the current epoch is exhausted, generate a new list for the next epoch
        if self.thread_id == 0:
            if self.ix >= len(self.indices):
                self.ix = 0
                
                # 1. Get all positive indices
                pos_indices_this_epoch = self.positive_indices

                # 2. Randomly sample negative indices
                neg_indices_this_epoch = np.random.choice(
                    self.negative_indices,
                    size=self.num_neg_to_sample,
                    replace=False
                ) if len(self.negative_indices) > 0 else []

                # 3. Combine and shuffle the indices for the new epoch
                combined_indices = np.concatenate([pos_indices_this_epoch, neg_indices_this_epoch]).astype(int)
                np.random.shuffle(combined_indices)
                self.indices = combined_indices
        
        # This part is from the original batchgenerators DataLoader
        idx = self.indices[self.ix:self.ix + self.batch_size]
        self.ix += self.batch_size
        return idx

# =================================================================================
# ==> STEP 2: The Custom Trainer uses the new DataLoader
# =================================================================================
class nnUNetTrainer_CustomSampler(nnUNetTrainer):

    def on_train_start(self):
        super().on_train_start()

        training_dataset_loader = self.dataloader_train.generator
        underlying_dataset = training_dataset_loader._data

        positive_indices = []
        negative_indices = []
        
        all_case_identifiers = underlying_dataset.identifiers
        
        for idx, key in enumerate(all_case_identifiers):
            properties_file = join(underlying_dataset.source_folder, key + ".pkl")
            properties = load_pickle(properties_file)

            # --- ROBUST FOREGROUND CHECK ---
            is_positive = False
            if 'class_locations' in properties:
                for class_label, locations in properties['class_locations'].items():
                    if class_label != 0 and len(locations) > 0:
                        is_positive = True
                        break
            
            if is_positive:
                positive_indices.append(idx)
            else:
                negative_indices.append(idx)

        self.print_to_log_file(f"Custom Sampler: Identified {len(positive_indices)} positive and {len(negative_indices)} negative training cases.")

        new_dataloader = CustomnnUNetDataLoader(
            underlying_dataset,
            self.batch_size,
            patch_size=self.configuration_manager.patch_size,
            final_patch_size=training_dataset_loader.final_patch_size,
            label_manager=self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=training_dataset_loader.transforms,
            positive_indices=positive_indices,
            negative_indices=negative_indices
        )

        # Re-wrap the new dataloader in the MultiThreadedAugmenter
        # THIS IS THE FINAL CORRECTION: use num_cached_per_worker instead of num_cached
        allowed_num_processes = get_allowed_n_proc_DA()
        self.dataloader_train = MultiThreadedAugmenter(
            new_dataloader,
            self.dataloader_train.transform,
            num_processes=allowed_num_processes,
            num_cached_per_worker=2, # Corrected argument
            seeds=None,
            pin_memory=self.device.type == 'cuda'
        )
