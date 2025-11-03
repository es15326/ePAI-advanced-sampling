from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# We define a small wrapper class to intercept the 'foreground_labels' property
class LabelManagerWrapper:
    def __init__(self, original_label_manager, lesion_label):
        self._original_label_manager = original_label_manager
        self._lesion_label = lesion_label

    @property
    def foreground_labels(self):
        # When anything asks for foreground_labels, we return ONLY the lesion label.
        return [self._lesion_label]

    def __getattr__(self, name):
        # For any other attribute, we pass the request to the original label manager.
        return getattr(self._original_label_manager, name)


class nnUNetTrainer_LesionOversample(nnUNetTrainer):
    """
    This custom trainer uses a wrapper around the LabelManager to ensure that
    for oversampling, ONLY the lesion class is used as a foreground target.
    """
    def get_dataloaders(self):
        lesion_label = 18
        
        # Store the original label manager
        original_manager = self.label_manager
        
        # Create an instance of our wrapper and temporarily replace the trainer's label_manager
        self.label_manager = LabelManagerWrapper(original_manager, lesion_label)
        
        print("#######################################################################")
        print("USING CUSTOM TRAINER: nnUNetTrainer_LesionOversample (Wrapper Version)")
        print(f"Forcing foreground oversampling to ONLY use label: {self.label_manager.foreground_labels}")
        print("#######################################################################")
        
        # Call the original get_dataloaders method. It will now use our wrapper.
        dl_tr, dl_val = super().get_dataloaders()
        
        # IMPORTANT: Restore the original label_manager after dataloaders are created.
        self.label_manager = original_manager
        
        return dl_tr, dl_val
