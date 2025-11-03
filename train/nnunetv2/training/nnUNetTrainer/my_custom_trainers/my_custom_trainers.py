import numpy as np
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

# You can add this class to the same file as your previous custom trainer.

class nnUNetTrainer_BalancedSampling(nnUNetTrainer):
    """
    This trainer implements a custom case-level sampling strategy for each epoch.
    It ensures that each epoch uses all positive samples and a randomly selected
    subset of negative samples, creating a balanced 1:1 dataset for each epoch.
    """

    def on_train_start(self):
        """
        This function is called once before the training loop starts.
        We use it to identify and store the lists of positive and negative training keys.
        """
        # First, run the standard on_train_start procedure.
        # This will create the dataloaders, which we need to access the dataset.
        super().on_train_start()

        self.print_to_log_file("Identifying positive and negative cases for balanced sampling...")

        # Access the training dataset from the dataloader
        dataset_tr = self.dataloader_train.generator.data

        self.positive_keys = []
        self.negative_keys = []

        # Iterate through all training cases for this fold
        for key in dataset_tr.identifiers:
            # We need to load the segmentation data to check if it's a positive or negative case.
            # load_case returns: data, seg, properties
            # We only need the segmentation `seg` here.
            _, seg, _, _ = dataset_tr.load_case(key)

            # A case is considered "positive" if its segmentation mask contains any label other than 0 (background).
            if np.any(seg > 0):
                self.positive_keys.append(key)
            else:
                self.negative_keys.append(key)

        self.print_to_log_file(f"Case identification complete.")
        self.print_to_log_file(f"Total positive cases found: {len(self.positive_keys)}")
        self.print_to_log_file(f"Total negative cases found: {len(self.negative_keys)}")

        if not self.positive_keys or not self.negative_keys:
            raise RuntimeError("Could not find both positive and negative samples. The balanced sampler will not work.")

    def on_train_epoch_start(self):
        """
        This function is called at the beginning of each training epoch.
        We override it to create and set our custom list of training keys.
        """
        # First, run the standard parent function to update learning rates etc.
        super().on_train_epoch_start()

        self.print_to_log_file("Creating balanced sample list for this epoch...")

        # Randomly select 17 negative keys.
        # `replace=False` ensures we don't pick the same negative case twice in one epoch.
        num_neg_to_sample = min(len(self.positive_keys), len(self.negative_keys))
        
        self.print_to_log_file(f"Sampling {num_neg_to_sample} negative cases to match {len(self.positive_keys)} positive cases.")

        selected_negative_keys = np.random.choice(self.negative_keys, size=num_neg_to_sample, replace=False).tolist()

        # Combine all positive keys with the randomly selected negative keys.
        current_epoch_keys = self.positive_keys + selected_negative_keys

        # The magic happens here: we overwrite the list of keys the dataloader will use for this epoch.
        self.dataloader_train.generator.indices = current_epoch_keys

        self.print_to_log_file(f"This epoch will train on {len(current_epoch_keys)} cases ({len(self.positive_keys)} positive, {len(selected_negative_keys)} negative).")
