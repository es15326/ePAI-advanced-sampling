# Filename: nnUNetDataLoader_Targeted_dynamic.py
import numpy as np
import torch

# --- Robust Import Strategy ---
try:
    # Attempt relative import first
    from .data_loader import nnUNetDataLoader
except ImportError:
    try:
        # Fallback to standard import path
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    except ImportError:
        raise ImportError("Could not import base nnUNetDataLoader.")
# --------------------------------------------

class nnUNetDataLoader_Targeted(nnUNetDataLoader):
    def __init__(self, *args, target_label_id=None, target_sampling_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_label_id = target_label_id
        self.target_sampling_ratio = target_sampling_ratio
        
        if self.target_label_id is not None:
             # Use print for visibility during initialization
             print(f"Targeted Oversampling Active in DataLoader. ID: {self.target_label_id}, Ratio: {self.target_sampling_ratio}.")

    # Robust implementation utilizing 'overwrite_class'.
    def get_bbox(self, data_shape, force_fg: bool, class_locations: dict, overwrite_class=None, annotated_classes_key=None):
        
        if annotated_classes_key is None:
            # Safely access the attribute if not passed
            annotated_classes_key = getattr(self, 'annotated_classes_key', None)

        # Check if targeted sampling should intervene.
        # We only intervene if force_fg is True AND no specific class override is already requested.
        if not force_fg or self.target_label_id is None or not class_locations or overwrite_class is not None:
            return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, annotated_classes_key)

        # --- Targeted Sampling Logic ---
        
        # 1. Check if the target label is present in the image
        target_present = (self.target_label_id in class_locations) and (len(class_locations[self.target_label_id]) > 0)

        # 2. Decide whether to force the target class
        # Use np.random.random() as it is correctly seeded per worker by the augmenter.
        if target_present and (np.random.random() < self.target_sampling_ratio):
            # Strategy: Force the target class by setting overwrite_class.
            # The base implementation handles the selection using this argument.
            return super().get_bbox(data_shape, True, class_locations, self.target_label_id, annotated_classes_key)

        # 3. Fallback: Use standard behavior.
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, annotated_classes_key)
