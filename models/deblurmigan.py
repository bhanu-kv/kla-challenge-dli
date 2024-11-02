import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# Define the generator (U-Net based with 900x900 output)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Bottleneck
        self.bottleneck = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),  # Input: 3 channels (RGB)
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=0)
        )
        self.fc1 = nn.Linear(529, 100)
        self.fc2 = nn.Linear(100, 1)# Fully connected layer to output a single scalar

    def forward(self, img):
        # Pass through convolutional layers
        out = self.model(img)
        # Flatten the output for the fully connected layer
        out = out.view(out.size(0), -1)  # Flatten to (batch_size, 512)
        out = self.fc1(out)  # Get scalar output
        out = self.fc2(out)  # Get scalar output
        out = torch.sigmoid(out)
        return out

