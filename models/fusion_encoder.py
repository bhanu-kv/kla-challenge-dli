import torch
import torch.nn as nn

class FusionEncoder(nn.Module):
    def __init__(self):
        super(FusionEncoder, self).__init__()
        
        # Initial convolution layers
        self.conv1 = nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1)  # 7 input channels -> 64 channels
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)  # 64 channels -> 128 channels
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)  # 128 channels -> 256 channels

        # Additional layers to deepen the network
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  # 512 channels
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)  # 256 channels
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)  # 128 channels
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)  # 64 channels

        # Final output layer to reduce to 3 channels
        self.output_conv = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        
        # Activation function
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, grayscale_img, rgb_img1, rgb_img2, rgb_img3, rgb_img4):
        # Concatenate the images along the channel dimension to get a (7, 900, 900) tensor
        x = torch.cat([grayscale_img, rgb_img1, rgb_img2, rgb_img3, rgb_img4], dim=1)  # (batch, 7, 900, 900)

        # Pass through convolutional layers
        x = self.relu(self.conv1(x))  # Maintains 900 x 900
        x = self.relu(self.conv2(x))  # Maintains 900 x 900
        x = self.relu(self.conv3(x))  # Maintains 900 x 900
        x = self.relu(self.conv4(x))  # Maintains 900 x 900
        x = self.relu(self.conv5(x))  # Maintains 900 x 900
        x = self.relu(self.conv6(x))  # Maintains 900 x 900
        x = self.relu(self.conv7(x))  # Maintains 900 x 900
        
        # Final layer to reduce to 3 channels (RGB output)
        fused_output = self.output_conv(x)  # (batch, 3, 900, 900)
        return fused_output