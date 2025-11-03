# Filename: nnUNetTrainer_MetaDynamic.py
import torch
import numpy as np
import os
from torch import autocast
import copy
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join, isfile

# Import the DynamicSampling trainer (using relative import)
try:
    # Ensure nnUNetTrainer_DynamicSampling.py is in the same directory
    from .nnUNetTrainer_DynamicSampling import nnUNetTrainer_DynamicSampling
except ImportError:
     print("ERROR: Cannot import nnUNetTrainer_DynamicSampling."); raise

# Import custom network (using relative import)
try:
    # Ensure unet_with_metadata.py is in the same directory
    from .unet_with_metadata import PlainConvUNet_Metadata
except ImportError:
    print("ERROR: Failed to import custom Metadata network (unet_with_metadata.py)."); raise

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
# We need this to determine the base dataset class dynamically
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


# ====================================================================================
# Dynamic Dataset Class Factory (Replaces the separate nnUNetDataset_Metadata.py file)
# ====================================================================================

def create_metadata_dataset_class(BaseDatasetClass):
    """
    Dynamically creates a Dataset class that inherits from the provided BaseDatasetClass
    and adds metadata loading functionality.
    """
    class DynamicMetadataDataset(BaseDatasetClass):
        def __init__(self, *args, **kwargs):
            # Call the constructor of the dynamically determined base class
            super().__init__(*args, **kwargs)
            
            # Determine the folder path robustly from arguments or instance attributes
            folder = kwargs.get('folder')
            if folder is None and hasattr(self, 'folder'):
                folder = self.folder
            elif folder is None and len(args) > 0 and isinstance(args[0], str):
                 folder = args[0]
            else:
                 folder = None

            self.metadata_dict = {}
            self.metadata_size = 0

            if folder:
                # Robustly find the root preprocessed folder (DatasetXXX)
                # The input folder might be a subdirectory (e.g., fold_X).
                base_dir = folder
                if not os.path.basename(base_dir).startswith("Dataset"):
                    original_dir = base_dir
                    # Navigate upwards
                    while base_dir and not os.path.basename(base_dir).startswith("Dataset"):
                        parent_dir = os.path.dirname(base_dir)
                        if parent_dir == base_dir:
                            base_dir = original_dir # Reset if search fails
                            break
                        base_dir = parent_dir

                metadata_file = join(base_dir, 'metadata_features.pkl')

                if isfile(metadata_file):
                    self.metadata_dict = load_pickle(metadata_file)
                    if len(self.metadata_dict) > 0:
                         first_key = next(iter(self.metadata_dict))
                         self.metadata_size = self.metadata_dict[first_key].shape[0]
                    print(f"Metadata loaded by Dynamic Dataset Class. Size: {self.metadata_size}")
                else:
                    print(f"WARNING: metadata_features.pkl not found at {metadata_file}. Proceeding without metadata.")

        def __getitem__(self, identifier_or_index):
            # Get the standard data/seg dictionary from the base class
            data_dict = super().__getitem__(identifier_or_index)
            
            # The base implementation ensures 'identifier' key is present in the returned dict.
            identifier = data_dict.get('identifier')

            if self.metadata_size > 0 and identifier:
                if identifier in self.metadata_dict:
                    metadata_vector = self.metadata_dict[identifier]
                else:
                    # Handle missing metadata with a zero vector
                    print(f"Warning: Metadata missing for {identifier}. Using zero vector.")
                    metadata_vector = np.zeros(self.metadata_size, dtype=np.float32)
                
                # Add the metadata vector to the dictionary
                data_dict['metadata'] = metadata_vector
            return data_dict
            
    return DynamicMetadataDataset

# ====================================================================================
# Trainer Implementation
# ====================================================================================

class nnUNetTrainer_MetaDynamic(nnUNetTrainer_DynamicSampling):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Initialized nnUNetTrainer_MetaDynamic (Metadata + Dynamic Sampling).")
        self.metadata_feature_size = 0
        self.DynamicMetadataDatasetClass = None # Placeholder for the dynamically created class

    def initialize(self):
        # --- DYNAMIC INHERITANCE LOGIC ---
        # This logic must run before calling super().initialize()
        
        # 1. Determine the base dataset class that nnU-Net intends to use.
        if self.dataset_class is None:
            try:
                # Infer the correct concrete class (e.g., standard Numpy or Blosc2)
                BaseDatasetClass = infer_dataset_class(self.preprocessed_dataset_folder)
                self.print_to_log_file(f"Inferred base dataset class: {BaseDatasetClass.__name__}")
            except Exception as e:
                self.print_to_log_file(f"ERROR: Failed to infer dataset class: {e}. Check preprocessed data.")
                raise
        else:
            BaseDatasetClass = self.dataset_class

        # 2. Create the custom dataset class dynamically using the factory
        self.DynamicMetadataDatasetClass = create_metadata_dataset_class(BaseDatasetClass)
        
        # 3. Set the dataset class override so the base initializer uses it
        self.dataset_class = self.DynamicMetadataDatasetClass
        # -------------------------------

        # Determine metadata size (using the dynamically created class)
        self.determine_metadata_size()
        
        # Proceed with standard initialization (which now uses the dynamic class)
        super().initialize()

    def determine_metadata_size(self):
        # Instantiate a temporary dataset using the dynamically created class.
        if self.DynamicMetadataDatasetClass is None:
            return

        try:
            # The constructor handles the file loading and size detection.
            # We pass the preprocessed folder; the dynamic constructor handles finding the metadata file.
            temp_dataset = self.DynamicMetadataDatasetClass(self.preprocessed_dataset_folder)
            self.metadata_feature_size = temp_dataset.metadata_size
            if self.metadata_feature_size > 0:
                self.print_to_log_file(f"Detected metadata feature vector size: {self.metadata_feature_size}")
        except Exception as e:
            # Catch potential issues during dataset instantiation (e.g. file access errors)
            self.print_to_log_file(f"WARNING: Could not determine metadata size during initialization: {e}. Proceeding without metadata.")
            self.metadata_feature_size = 0

    def get_network(self) -> torch.nn.Module:
        # Overrides to instantiate the custom network architecture.
        if self.metadata_feature_size == 0:
            # If no metadata, use the standard network initialization (call grandparent's method)
            return super(nnUNetTrainer_DynamicSampling, self).get_network()

        self.print_to_log_file(f"Initializing PlainConvUNet_Metadata with metadata_vector_size={self.metadata_feature_size}.")
        
        # Inject the custom network class and the metadata size into the configuration
        custom_arch_config = copy.deepcopy(self.configuration_manager.architecture)
        
        custom_arch_config['network_class'] = PlainConvUNet_Metadata
        custom_arch_config['arch_kwargs']['metadata_vector_size'] = self.metadata_feature_size

        # Use the utility function to build the network
        network = get_network_from_plans(
            self.plans_manager, custom_arch_config,
            self.configuration_manager.normalization_schemes,
            self.configuration_manager.num_modalities,
            self.label_manager.num_segmentation_heads,
            self.enable_deep_supervision
        )
        return network.to(self.device)

    def run_iteration(self, data_dict: dict, do_backprop: bool = True,
                      run_online_evaluation: bool = False):
        """
        Overrides the core training/validation loop to handle metadata passing.
        """
        
        # --- Extract data, target, and metadata ---
        data = data_dict['data']
        target = data_dict['target']
        metadata_vector = data_dict.get('metadata')

        # Move data and target to device
        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            
        # Handle metadata tensor
        if self.metadata_feature_size > 0:
            if metadata_vector is not None:
                # Convert numpy (from dataloader) to torch tensor
                if isinstance(metadata_vector, np.ndarray):
                    metadata_vector = torch.from_numpy(metadata_vector)
                
                # Ensure correct batch dimension if it was squeezed (e.g. batch size 1 during validation)
                if metadata_vector.ndim == 1 and data.shape[0] == 1:
                     metadata_vector = metadata_vector.unsqueeze(0)

                metadata_vector = metadata_vector.to(self.device, non_blocking=True)
            else:
                # Fallback if metadata is expected but missing in the batch
                self.print_to_log_file("Warning: Metadata expected but missing in batch. Using zero vector fallback.")
                metadata_vector = torch.zeros((data.shape[0], self.metadata_feature_size), dtype=data.dtype, device=self.device)

        # --------------------------------------------------

        self.optimizer.zero_grad()

        with autocast(self.device.type, enabled=self.enable_amp):
            # MODIFIED: Pass metadata_vector to the network
            if self.metadata_feature_size > 0:
                output = self.network(data, metadata=metadata_vector)
            else:
                output = self.network(data)
            
            # Cleanup
            if 'metadata_vector' in locals():
                del metadata_vector
            del data
            
            l = self.loss(output, target)

        # (Standard backpropagation and evaluation logic)
        if do_backprop:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        return l.detach().cpu().numpy()
