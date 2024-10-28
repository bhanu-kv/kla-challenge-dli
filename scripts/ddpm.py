import torch
import torch.nn as nn

# Define the UNet architecture
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Define encoder layers
        self.enc1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, stride=2)  # Downsample
        self.dec1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)  # Upsample
        self.dec2 = nn.Conv2d(64, 3, kernel_size=1)  # Output layer

    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pooled = self.pool(enc1)

        dec1 = self.dec1(enc1_pooled)

        # Print shapes for debugging
        print(f"enc1 shape: {enc1.shape}, dec1 shape: {dec1.shape}")

        # Ensure dec1 matches the size of enc1 for skip connection
        if dec1.shape[2:] != enc1.shape[2:]:
            print(f"Resizing dec1 from {dec1.shape} to {enc1.shape}")
            dec1 = nn.functional.interpolate(dec1, size=enc1.shape[2:], mode='bilinear', align_corners=True)

        # Skip connection
        dec2 = dec1 + enc1
        dec2 = self.dec2(dec2)
        return dec2


# Define the DDPM class
class DDPM:
    def __init__(self, model, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.alpha = torch.linspace(1, 0.001, timesteps)
        self.alpha = self.alpha.to(torch.float32)
        self.beta = 1 - self.alpha

    def forward_diffusion(self, x_0):
        noise = torch.randn_like(x_0)
        x_t = x_0.clone()
        for t in range(self.timesteps):
            x_t = self.alpha[t] * x_0 + (1 - self.alpha[t]) * noise
            yield x_t

    def reverse_diffusion(self, x_t):
        for t in reversed(range(self.timesteps)):
            with torch.no_grad():
                x_t = self.model(x_t)
        return x_t
