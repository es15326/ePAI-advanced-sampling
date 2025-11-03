# Filename: nnUNetTrainer_DynamicSampling_targeted.py
import torch
import numpy as np
# The base nnUNetTrainer is patched dynamically by the Curriculum trainer.
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import List

class nnUNetTrainer_DynamicSampling_targeted(nnUNetTrainer):
    """
    Implementation of Dynamic Sampling (DS) strategy focusing on specific labels.
    """

    # Use **kwargs for flexibility. This is safe as it's an intermediate class.
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, **kwargs):

        # Pass all arguments up the chain.
        super().__init__(plans, configuration, fold, dataset_json, **kwargs)

        # --- Dynamic Sampling Configuration ---
        self.enable_custom_sampling = True
        
        # Define the target label ID (Adjust as needed for the dataset)
        self.target_label_id = 2 

        # Define the sampling strategy
        self.negative_to_positive_ratio = 1.0 
        self.num_negative_samples_per_epoch = 50 

        print(f"Initialized nnUNetTrainer_DynamicSampling_targeted.")
        print(f"  DS Config: Target={self.target_label_id}, N:P Ratio={self.negative_to_positive_ratio}, Max Neg={self.num_negative_samples_per_epoch}")


    def classify_keys(self, dataset) -> tuple[List[str], List[str]]:
        """
        Classifies dataset keys based on the presence of the target label using metadata.
        """
        positive_keys = []
        negative_keys = []

        if not dataset:
             return positive_keys, negative_keys

        # Robustly find the metadata key for class presence
        try:
            first_key = next(iter(dataset.keys()))
            props = dataset.load_properties(first_key)
        except StopIteration:
            return positive_keys, negative_keys

        annotated_classes_key = None
        if 'annotated_classes' in props:
            annotated_classes_key = 'annotated_classes'
        elif 'class_presence' in props:
             annotated_classes_key = 'class_presence'
        
        if annotated_classes_key is None:
             print("ERROR: Could not find annotation metadata. Cannot perform Dynamic Sampling.")
             return list(dataset.keys()), []

        # Perform classification
        for k in dataset.keys():
            props = dataset.load_properties(k)
            annotated_classes = props.get(annotated_classes_key, [])
            
            try:
                present_labels = [int(label) for label in annotated_classes if label is not None]
            except (ValueError, TypeError):
                 print(f"WARNING: Invalid label format found for key {k}. Skipping.")
                 continue

            if self.target_label_id in present_labels:
                positive_keys.append(k)
            else:
                negative_keys.append(k)
        
        return positive_keys, negative_keys

    def on_train_epoch_start(self):
        """
        Implements the Dynamic Sampling logic (Global Shuffle) at the start of each epoch.
        """
        # Standard procedures
        if hasattr(self, 'network') and callable(getattr(self.network, 'train', None)):
            self.network.train()
            
        if self.lr_scheduler is not None:
            try:
                self.lr_scheduler.step(self.current_epoch)
            except TypeError:
                pass # Handle schedulers that don't take epoch argument

        if not getattr(self, 'enable_custom_sampling', False):
            return
        
        log_func = getattr(self, 'print_to_log_file', print)

        if not hasattr(self, 'positive_keys_tr') or not hasattr(self, 'negative_keys_tr'):
             log_func("ERROR: Missing positive/negative keys.")
             return

        # Ensure RNG is the modern Generator
        if not hasattr(self, 'RNG') or not isinstance(self.RNG, np.random.Generator):
             self.RNG = np.random.default_rng(getattr(self, 'fold', 0))

        positive_keys = list(self.positive_keys_tr)
        negative_keys = list(self.negative_keys_tr)

        # 1. Shuffle Positive Keys
        self.RNG.shuffle(positive_keys)
        
        # 2. Determine Number of Negative Samples
        num_pos = len(positive_keys)
        
        ratio = getattr(self, 'negative_to_positive_ratio', 1.0)
        num_neg_to_sample = int(np.round(num_pos * ratio))
        
        # Constrain by fixed limit if defined
        if hasattr(self, 'num_negative_samples_per_epoch'):
             max_neg = self.num_negative_samples_per_epoch
             if isinstance(max_neg, (int, float)) and not isinstance(max_neg, bool):
                 if num_neg_to_sample > max_neg:
                      num_neg_to_sample = int(max_neg)

        # 3. Sample Negative Keys
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

        # 4. Combine and Global Shuffle
        keys_for_epoch = positive_keys + sampled_negative_keys
        self.RNG.shuffle(keys_for_epoch)

        # 5. Update the data loader
        log_func(f"Epoch {self.current_epoch} DS (Global Shuffle): {len(keys_for_epoch)} cases ({num_pos} pos, {len(sampled_negative_keys)} neg).")

        if not self._update_dataloader_indices_robust(keys_for_epoch):
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
                if not self._update_dataloader_indices_robust(keys_for_epoch):
                     raise RuntimeError("Failed to configure DataLoader indices on distributed worker.")

    def _update_dataloader_indices_robust(self, keys_for_epoch):
        # Helper function to robustly update indices
        def update_loader(loader):
            if loader is None: return False
            if hasattr(loader, 'indices'):
                loader.indices = keys_for_epoch
                return True
            if hasattr(loader, 'data') and hasattr(loader.data, 'indices'):
                 loader.data.indices = keys_for_epoch
                 return True
            if hasattr(loader, 'dataset') and hasattr(loader.dataset, 'indices'):
                loader.dataset.indices = keys_for_epoch
                return True
            return False

        # Identify and update the raw loader
        # Use getattr safely as attributes might not be initialized depending on lifecycle
        raw_loader = getattr(self, 'dl_tr_raw', getattr(self, 'dl_tr', getattr(self, 'dataloader_train', None)))
        updated_raw = update_loader(raw_loader)
        
        # Identify and update the augmenter's internal loader (if exists)
        tr_gen_loader = getattr(getattr(self, 'tr_gen', None), 'data_loader', None)
        updated_gen = update_loader(tr_gen_loader)
        
        return updated_raw or updated_gen
