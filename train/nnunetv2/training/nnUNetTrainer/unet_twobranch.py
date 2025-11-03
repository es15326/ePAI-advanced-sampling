# Filename: unet_twobranch.py
import torch
from torch import nn
try:
    # We only need the base U-Net class now. The problematic helper imports are removed.
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except ImportError:
    print("ERROR: dynamic_network_architectures not found. Check installation."); raise

class PlainConvUNet_TwoBranch(PlainConvUNet):
    def __init__(self, *args, metadata_vector_size: int = 0, **kwargs):
        # Initialize the base U-Net first.
        super().__init__(*args, **kwargs)
        
        self.metadata_vector_size = metadata_vector_size
        self.metadata_mlp = None
        self.mlp_output_size = 0

        if self.metadata_vector_size > 0:
            # Pass 'self' so the function can access the U-Net's configuration
            self.initialize_two_branch_architecture(self)

    def _get_activation_from_unet(self, unet_instance):
        """
        Helper to robustly retrieve the activation function class and kwargs 
        from the U-Net instance without relying on external helper functions.
        """
        # Access the configuration stored by PlainConvUNet during its initialization
        # These attributes ('nonlin', 'nonlin_kwargs') are set in PlainConvUNet.__init__
        nonlin_name = getattr(unet_instance, 'nonlin', 'torch.nn.LeakyReLU')
        nonlin_kwargs = getattr(unet_instance, 'nonlin_kwargs', {'inplace': True})

        # Resolve the name string (e.g., 'torch.nn.ReLU') to the actual class (torch.nn.ReLU)
        if isinstance(nonlin_name, str):
            # Handle namespaced strings
            if nonlin_name.startswith('torch.nn.'):
                nonlin_name = nonlin_name[9:]
            
            if hasattr(torch.nn, nonlin_name):
                NonlinLayer = getattr(torch.nn, nonlin_name)
            else:
                # Fallback if custom activation is used but not resolvable
                print(f"Warning: Could not resolve activation {nonlin_name}. Falling back to LeakyReLU.")
                NonlinLayer = torch.nn.LeakyReLU
                nonlin_kwargs = {'inplace': True}
        elif callable(nonlin_name):
             # If it's already a class/callable
             NonlinLayer = nonlin_name
        else:
            # Final safety fallback
            NonlinLayer = torch.nn.LeakyReLU
            nonlin_kwargs = {'inplace': True}
            
        return NonlinLayer, nonlin_kwargs


    def initialize_two_branch_architecture(self, unet_instance):
        # --- 1. Define the Metadata Branch (MLP) ---
        
        # Configuration for the MLP
        mlp_hidden_size = 128
        self.mlp_output_size = 64 

        # Determine Normalization Layer for MLP. LayerNorm is robust for MLPs.
        try:
            NormLayer = nn.LayerNorm
            # LayerNorm requires the shape of the features it normalizes
            norm_kwargs_mlp = {'normalized_shape': mlp_hidden_size}
        except AttributeError:
            # Fallback for very old PyTorch versions if LayerNorm is missing
            NormLayer = nn.BatchNorm1d
            norm_kwargs_mlp = {}

        # Get the Activation function used in the U-Net (using the new internal helper)
        NonlinLayer, nonlin_kwargs = self._get_activation_from_unet(unet_instance)

        self.metadata_mlp = nn.Sequential(
            nn.Linear(self.metadata_vector_size, mlp_hidden_size),
            NormLayer(**norm_kwargs_mlp),
            NonlinLayer(**nonlin_kwargs),
            nn.Linear(mlp_hidden_size, self.mlp_output_size)
        )
        print(f"Metadata MLP initialized: Input {self.metadata_vector_size} -> Hidden {mlp_hidden_size} -> Output {self.mlp_output_size}")
        print(f"MLP Activation: {NonlinLayer.__name__}, Norm: {NormLayer.__name__}")

        # --- 2. Adapt the Decoder for Fusion ---
        
        if not hasattr(self.decoder, 'transp_convs') or len(self.decoder.transp_convs) == 0:
             print("ERROR: Architecture mismatch. Cannot find decoder transp_convs."); return

        transp_conv_layer = self.decoder.transp_convs[0]
        original_in_channels = transp_conv_layer.in_channels
        new_in_channels = original_in_channels + self.mlp_output_size

        # Create a replacement layer
        # Determine the type (ConvTranspose2d or ConvTranspose3d) dynamically
        ConvTransposeClass = type(transp_conv_layer) 
        new_transp_conv_layer = ConvTransposeClass(
            new_in_channels,
            transp_conv_layer.out_channels,
            kernel_size=transp_conv_layer.kernel_size,
            stride=transp_conv_layer.stride,
            padding=transp_conv_layer.padding,
            output_padding=transp_conv_layer.output_padding,
            bias=transp_conv_layer.bias is not None,
        )

        # Initialize weights (Kaiming)
        # Determine the appropriate nonlinearity parameter for initialization
        init_mode = 'leaky_relu' if 'LeakyReLU' in NonlinLayer.__name__ else 'relu'
        nn.init.kaiming_normal_(new_transp_conv_layer.weight, mode='fan_in', nonlinearity=init_mode)
        
        # Copy existing weights if possible (useful if resuming a partially trained standard U-Net)
        try:
            with torch.no_grad():
                # Copy weights for the original image feature channels
                new_transp_conv_layer.weight.data[:original_in_channels, ...] = transp_conv_layer.weight.data
                if transp_conv_layer.bias is not None:
                    new_transp_conv_layer.bias.data = transp_conv_layer.bias.data
        except Exception as e:
             print(f"Note: Could not copy weights from original layer ({e}). Relying on Kaiming initialization.")


        # Replace the layer in the decoder module list
        self.decoder.transp_convs[0] = new_transp_conv_layer
        print(f"Two-Branch Architecture Active (Bottleneck Fusion).")
        print(f"  Decoder Adaptation: First upsampling input channels changed from {original_in_channels} to {new_in_channels}.")

    # (The forward method remains the same)
    def forward(self, x, metadata=None):
        if self.metadata_vector_size > 0 and (metadata is None or metadata.numel() == 0):
             raise ValueError("Network expects metadata input but received None or empty tensor.")
        
        # 1. Image Branch (Encoder)
        skips = self.encoder(x)
        bottleneck_features = skips[-1]
        
        if self.metadata_vector_size > 0 and self.metadata_mlp is not None:
            # 2. Metadata Branch (MLP)
            # The trainer ensures metadata is 2D (B, F), so we can pass it directly.
            processed_metadata = self.metadata_mlp(metadata)
            
            # 3. Fusion (Tiling and Concatenation)
            spatial_dims = bottleneck_features.shape[2:]
            # Reshape: (B, F_mlp) -> (B, F_mlp, 1, 1, [1])
            metadata_expanded = processed_metadata.view(processed_metadata.size(0), processed_metadata.size(1), *([1] * len(spatial_dims)))
            # Tile: (B, F_mlp, 1, 1, [1]) -> (B, F_mlp, X, Y, [Z])
            metadata_tiled = metadata_expanded.repeat(1, 1, *spatial_dims)
            
            # Concatenate
            concatenated_bottleneck = torch.cat((bottleneck_features, metadata_tiled), dim=1)
            
            # Replace the bottleneck features with the fused version
            skips[-1] = concatenated_bottleneck
        
        # 4. Decoder
        return self.decoder(skips)
