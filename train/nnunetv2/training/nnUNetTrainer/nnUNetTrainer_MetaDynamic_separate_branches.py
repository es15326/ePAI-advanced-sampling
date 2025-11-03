# Filename: nnUNetTrainer_MetaDynamic.py
import torch
import numpy as np
import os
from torch import autocast
import copy
from batchgenerators.utilities.file_and_folder_operations import load_pickle, join, isfile

# Import the DynamicSampling trainer (using relative import)
try:
    from .nnUNetTrainer_DynamicSampling import nnUNetTrainer_DynamicSampling
except ImportError:
     print("ERROR: Cannot import nnUNetTrainer_DynamicSampling."); raise

# --- MODIFICATION START: Import the new TwoBranch network ---
try:
    # Import from the new architecture file: unet_twobranch.py
    from .unet_twobranch import PlainConvUNet_TwoBranch
except ImportError:
    print("ERROR: Failed to import custom TwoBranch network (unet_twobranch.py)."); raise
# --- MODIFICATION END ---

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class


# ====================================================================================
# Dynamic Dataset Class Factory 
# (No changes needed in this section compared to the previous implementation)
# ====================================================================================
def create_metadata_dataset_class(BaseDatasetClass):
    class DynamicMetadataDataset(BaseDatasetClass):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # (Metadata loading logic remains the same)
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
                base_dir = folder
                if not os.path.basename(base_dir).startswith("Dataset"):
                    original_dir = base_dir
                    while base_dir and not os.path.basename(base_dir).startswith("Dataset"):
                        parent_dir = os.path.dirname(base_dir)
                        if parent_dir == base_dir:
                            base_dir = original_dir
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
                    print(f"WARNING: metadata_features.pkl not found at {metadata_file}.")

        def __getitem__(self, identifier_or_index):
            data_dict = super().__getitem__(identifier_or_index)
            identifier = data_dict.get('identifier')

            if self.metadata_size > 0 and identifier:
                if identifier in self.metadata_dict:
                    metadata_vector = self.metadata_dict[identifier]
                else:
                    print(f"Warning: Metadata missing for {identifier}. Using zero vector.")
                    metadata_vector = np.zeros(self.metadata_size, dtype=np.float32)
                
                data_dict['metadata'] = metadata_vector
            return data_dict
            
    return DynamicMetadataDataset

# ====================================================================================
# Trainer Implementation
# ====================================================================================

class nnUNetTrainer_MetaDynamic_separate_branches(nnUNetTrainer_DynamicSampling):
    # (No changes needed in __init__, initialize, or determine_metadata_size)
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Initialized nnUNetTrainer_MetaDynamic (Two-Branch Architecture).")
        self.metadata_feature_size = 0
        self.DynamicMetadataDatasetClass = None 

    def initialize(self):
        # (Dynamic inheritance logic remains the same)
        if self.dataset_class is None:
            try:
                BaseDatasetClass = infer_dataset_class(self.preprocessed_dataset_folder)
                self.print_to_log_file(f"Inferred base dataset class: {BaseDatasetClass.__name__}")
            except Exception as e:
                self.print_to_log_file(f"ERROR: Failed to infer dataset class: {e}."); raise
        else:
            BaseDatasetClass = self.dataset_class

        self.DynamicMetadataDatasetClass = create_metadata_dataset_class(BaseDatasetClass)
        self.dataset_class = self.DynamicMetadataDatasetClass
        
        self.determine_metadata_size()
        super().initialize()

    def determine_metadata_size(self):
        if self.DynamicMetadataDatasetClass is None: return
        try:
            temp_dataset = self.DynamicMetadataDatasetClass(self.preprocessed_dataset_folder)
            self.metadata_feature_size = temp_dataset.metadata_size
            if self.metadata_feature_size > 0:
                self.print_to_log_file(f"Detected metadata feature vector size: {self.metadata_feature_size}")
        except Exception as e:
            self.print_to_log_file(f"WARNING: Could not determine metadata size: {e}.")
            self.metadata_feature_size = 0

    # --- MODIFICATION START: Update get_network ---
    def get_network(self) -> torch.nn.Module:
        if self.metadata_feature_size == 0:
            return super(nnUNetTrainer_DynamicSampling, self).get_network()

        self.print_to_log_file(f"Initializing PlainConvUNet_TwoBranch.")
        
        custom_arch_config = copy.deepcopy(self.configuration_manager.architecture)
        
        # Use the new TwoBranch architecture
        custom_arch_config['network_class'] = PlainConvUNet_TwoBranch
        
        # Add the custom argument required by the network's constructor
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
    # --- MODIFICATION END ---

    # (No changes needed in run_iteration)
    def run_iteration(self, data_dict: dict, do_backprop: bool = True,
                      run_online_evaluation: bool = False):
        # (Implementation remains the same, as it already handles passing metadata)
        data = data_dict['data']
        target = data_dict['target']
        metadata_vector = data_dict.get('metadata')

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)
            
        if self.metadata_feature_size > 0:
            if metadata_vector is not None:
                if isinstance(metadata_vector, np.ndarray):
                    metadata_vector = torch.from_numpy(metadata_vector)
                
                if metadata_vector.ndim == 1 and data.shape[0] == 1:
                     metadata_vector = metadata_vector.unsqueeze(0)

                metadata_vector = metadata_vector.to(self.device, non_blocking=True)
            else:
                self.print_to_log_file("Warning: Metadata expected but missing in batch. Using zero vector fallback.")
                metadata_vector = torch.zeros((data.shape[0], self.metadata_feature_size), dtype=data.dtype, device=self.device)

        self.optimizer.zero_grad()

        with autocast(self.device.type, enabled=self.enable_amp):
            if self.metadata_feature_size > 0:
                output = self.network(data, metadata=metadata_vector)
            else:
                output = self.network(data)
            
            if 'metadata_vector' in locals():
                del metadata_vector
            del data
            
            l = self.loss(output, target)

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
