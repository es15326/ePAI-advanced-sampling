# Filename: nnUNetDataLoader_Targeted.py
import numpy as np
import torch # Required for type hints compatibility

# --- Robust Import Strategy ---
# Attempt relative import first, fallback to standard path.
try:
    from .data_loader import nnUNetDataLoader
except ImportError:
    try:
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
    except ImportError:
        raise ImportError("Could not import base nnUNetDataLoader. Check installation and file placement.")
# --------------------------------------------

class nnUNetDataLoader_Targeted(nnUNetDataLoader):
    def __init__(self, *args, target_label_id=None, target_sampling_ratio=0.75, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_label_id = target_label_id
        self.target_sampling_ratio = target_sampling_ratio
        
        if self.target_label_id is not None:
             print(f"Targeted Oversampling Active. ID: {self.target_label_id}, Ratio: {self.target_sampling_ratio}.")

    # Implementation utilizing 'overwrite_class' for robustness.
    def get_bbox(self, data_shape, force_fg: bool, class_locations: dict, overwrite_class=None, annotated_classes_key=None):
        
        # Ensure annotated_classes_key is available internally if not passed (robustness)
        if annotated_classes_key is None:
            # Use getattr for safe access
            annotated_classes_key = getattr(self, 'annotated_classes_key', None)

        # Check if targeted sampling should intervene.
        # We only intervene if force_fg is True AND no specific class override is already requested.
        if not force_fg or self.target_label_id is None or not class_locations or overwrite_class is not None:
            # If not active, or if a class is already forced, call the parent implementation directly.
            return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, annotated_classes_key)

        # --- Targeted Sampling Logic ---
        
        # 1. Check if the target label is present in the image
        target_present = (self.target_label_id in class_locations) and (len(class_locations[self.target_label_id]) > 0)

        # 2. Decide whether to force the target class based on the defined ratio
        # FIX: Use the global np.random instance (Resolves AttributeError: 'RNG'/'generator').
        # The NonDetMultiThreadedAugmenter ensures this global state is seeded correctly per worker.
        if target_present and (np.random.random() < self.target_sampling_ratio):
            # Strategy: Force the target class by setting overwrite_class to the target ID.
            # The base implementation handles the selection using this argument.
            
            # We ensure force_fg=True here as we are forcing the foreground.
            # We pass the target_label_id as the new overwrite_class.
            return super().get_bbox(data_shape, True, class_locations, self.target_label_id, annotated_classes_key)

        # 3. Fallback: If target was not present or not selected by the ratio, use standard behavior.
        # Call the parent implementation with the original arguments (overwrite_class remains None).
        return super().get_bbox(data_shape, force_fg, class_locations, overwrite_class, annotated_classes_key)
