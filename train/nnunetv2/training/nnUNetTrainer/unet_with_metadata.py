# Filename: unet_with_metadata.py
import torch
from torch import nn
try:
    from dynamic_network_architectures.architectures.unet import PlainConvUNet
except ImportError:
    print("ERROR: dynamic_network_architectures not found."); raise

class PlainConvUNet_Metadata(PlainConvUNet):
    def __init__(self, *args, metadata_vector_size: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata_vector_size = metadata_vector_size

        if self.metadata_vector_size > 0:
            self.inject_metadata_architecture()

    def inject_metadata_architecture(self):
        # Modify the first transposed convolution layer in the decoder.
        if not hasattr(self.decoder, 'transp_convs') or len(self.decoder.transp_convs) == 0:
             print("ERROR: Architecture mismatch. Cannot find decoder transp_convs."); return

        # Access the layer that consumes the bottleneck output
        transp_conv_layer = self.decoder.transp_convs[0]
        in_channels = transp_conv_layer.in_channels
        new_in_channels = in_channels + self.metadata_vector_size

        # Create a replacement layer
        ConvTransposeClass = type(transp_conv_layer) # Handles ConvTranspose2d or 3d
        new_transp_conv_layer = ConvTransposeClass(
            new_in_channels,
            transp_conv_layer.out_channels,
            kernel_size=transp_conv_layer.kernel_size,
            stride=transp_conv_layer.stride,
            padding=transp_conv_layer.padding,
            output_padding=transp_conv_layer.output_padding,
            bias=transp_conv_layer.bias is not None,
            dilation=transp_conv_layer.dilation,
            groups=transp_conv_layer.groups
        )

        # Initialize weights (Kaiming) and copy existing weights
        nn.init.kaiming_normal_(new_transp_conv_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        with torch.no_grad():
            # Copy weights for the original image feature channels
            new_transp_conv_layer.weight[:in_channels, ...] = transp_conv_layer.weight
            if transp_conv_layer.bias is not None:
                new_transp_conv_layer.bias.data = transp_conv_layer.bias.data

        # Replace the layer
        self.decoder.transp_convs[0] = new_transp_conv_layer
        print(f"Metadata Injection Active: First upsampling input channels increased from {in_channels} to {new_in_channels}.")

    def forward(self, x, metadata=None):
        if self.metadata_vector_size > 0 and (metadata is None or metadata.numel() == 0):
             raise ValueError("Network expects metadata input but received None or empty tensor.")
        
        # Standard encoder pass
        # skips[-1] contains the bottleneck features
        skips = self.encoder(x)
        
        if self.metadata_vector_size > 0:
            bottleneck_features = skips[-1]
            
            # Expand and Tile metadata (B, F) to match spatial dimensions (B, C, X, Y, [Z])
            spatial_dims = bottleneck_features.shape[2:]
            # Reshape: (B, F) -> (B, F, 1, 1, [1])
            metadata_expanded = metadata.view(metadata.size(0), metadata.size(1), *([1] * len(spatial_dims)))
            # Tile: (B, F, 1, 1, [1]) -> (B, F, X, Y, [Z])
            metadata_tiled = metadata_expanded.repeat(1, 1, *spatial_dims)
            
            # Concatenate
            concatenated_bottleneck = torch.cat((bottleneck_features, metadata_tiled), dim=1)
            
            # Replace the bottleneck features with the concatenated version
            skips[-1] = concatenated_bottleneck
        
        # Standard decoder pass (uses the modified skips[-1])
        return self.decoder(skips)
