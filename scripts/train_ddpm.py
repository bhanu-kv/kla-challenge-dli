#!/usr/bin/env python3

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/hcl/bhanu/project')

import sys
import os
from PIL import Image
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from models.ddpm import DDPM, UNet  # Ensure you have defined your DDPM and UNet in models.ddpm

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        for object_name in os.listdir(self.root_dir):
            object_path = os.path.join(self.root_dir, object_name, 'Train')
            degraded_path = os.path.join(object_path, 'Degraded_image')
            mask_path = os.path.join(object_path, 'Defect_mask')
            clean_path = os.path.join(object_path, 'GT_clean_image')
            for defect_type in os.listdir(degraded_path):
                degraded_defect_dir = os.path.join(degraded_path, defect_type)
                mask_defect_dir = os.path.join(mask_path, defect_type)
                clean_defect_dir = os.path.join(clean_path, defect_type)
                for img_name in os.listdir(degraded_defect_dir):
                    degraded_img = os.path.join(degraded_defect_dir, img_name)
                    mask_img = os.path.join(mask_defect_dir, img_name.replace(".png", "_mask.png"))
                    clean_img = os.path.join(clean_defect_dir, img_name)
                    if os.path.exists(degraded_img) and os.path.exists(clean_img):
                        triplets.append((degraded_img, mask_img, clean_img))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        degraded_img_path, mask_img_path, clean_img_path = self.triplets[idx]
        degraded_img = Image.open(degraded_img_path).convert("RGB")
        mask_img = Image.open(mask_img_path).convert("RGB")
        clean_img = Image.open(clean_img_path).convert("RGB")
        if self.transform:
            degraded_img = self.transform(degraded_img)
            mask_img = self.transform(mask_img)
            clean_img = self.transform(clean_img)
        return degraded_img, mask_img, clean_img

# Load data loader from pickle file
filename = "./dataloader_train_revisedv1.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    psnr_ls = []
    psnr_worst = []
    
    # Training the DDPM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)  # Move UNet model to device
    ddpm = DDPM(model)  # Ensure DDPM and its internal model are on the device
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Directory to save model weights
    weights_dir = "../model_weights/ddpm_weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Training loop
    num_epochs = 10

    # Assume `model` and `optimizer` are already defined and initialized
    for epoch in range(num_epochs):
        optimizer.zero_grad()  # Reset gradients
        for i, (degraded_imgs, mask_imgs, clean_imgs) in enumerate(data_loader):
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)

            # Check if tensors require gradients
            if not degraded_imgs.requires_grad:
                degraded_imgs.requires_grad_()
            if not clean_imgs.requires_grad:
                clean_imgs.requires_grad_()

            # Forward diffusion process
            denoised_imgs = []
            for output in ddpm.forward_diffusion(clean_imgs):
                denoised_imgs.append(output)
            denoised_imgs = torch.stack(denoised_imgs)[-1]

            # Loss computation
            loss = nn.MSELoss()(denoised_imgs, clean_imgs)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Clear references
            del degraded_imgs, clean_imgs, denoised_imgs  
            torch.cuda.empty_cache()  # Clear unused memory
            print(i)
            
        # Save model weights after each epoch
        torch.save(model.state_dict(), os.path.join(weights_dir, f"ddpm_epoch_{epoch+1}.pth"))
        print(f"Epoch [{epoch+1}/{num_epochs}], Total Loss: {loss.item():.4f}")

