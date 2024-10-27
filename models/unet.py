import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDenoiser(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256]):
        super(UNetDenoiser, self).__init__()
        
        # Define the encoder
        self.encoder = nn.ModuleList()
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature
            
        # Define the decoder
        self.decoder = nn.ModuleList()
        for feature in reversed(features):
            self.decoder.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.decoder.append(self._conv_block(feature * 2, feature))
            
        # Bottleneck (middle part)
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)
        
        # Final output layer
        self.output_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)
        
    def forward(self, x):
        skip_connections = []
        for enc_layer in self.encoder:
            x = enc_layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # Reverse for decoding
        
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)  # Up-sample
            skip_connection = skip_connections[i // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])  # Ensure matching dimensions
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i + 1](x)  # Apply conv block
            
        return self.output_layer(x)
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
