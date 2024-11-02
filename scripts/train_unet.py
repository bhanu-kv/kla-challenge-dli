#!/usr/bin/env python3

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/hcl/bhanu/project')

import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from models.unet import UNetDenoiser


class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        for object_name in os.listdir(self.root_dir):
            #print(object_name)
            object_path = os.path.join(self.root_dir, object_name, 'Train')
            degraded_path = os.path.join(object_path, 'Degraded_image')
            mask_path = os.path.join(object_path, 'Defect_mask')
            clean_path = os.path.join(object_path, 'GT_clean_image')
            # Iterate through each defect type (broken_large, broken_small, etc.)
            for defect_type in os.listdir(degraded_path):
                #print(defect_type)
                degraded_defect_dir = os.path.join(degraded_path, defect_type)
                mask_defect_dir = os.path.join(mask_path, defect_type)
                clean_defect_dir = os.path.join(clean_path, defect_type)
                # Match image triplets by name in each defect folder
                for img_name in os.listdir(degraded_defect_dir):
                    degraded_img = os.path.join(degraded_defect_dir, img_name)
                    mask_img = os.path.join(mask_defect_dir, img_name.replace(".png", "_mask.png"))
                    clean_img = os.path.join(clean_defect_dir, img_name)
                    # Only add the triplet if all three files exist
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
        # Apply transforms, if any
        if self.transform:
            degraded_img = self.transform(degraded_img)
            mask_img = self.transform(mask_img)
            clean_img = self.transform(clean_img)
        return degraded_img, mask_img, clean_img
    
filename = "./dataloader_train_revisedv1.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    psnr_ls = []
    psnr_worst = []
    
    # Assuming UNetDenoiser and train_loader are already defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # test_weights_path = "../model_weights/unet_weights/fusion_epoch_8.pth"  # Load final epoch weights
    model = UNetDenoiser().to(device)
    # model.load_state_dict(torch.load(test_weights_path))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Directory to save model weights
    weights_dir = "../model_weights/unet_deep_weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Training loop
    num_epochs = 50

    for epoch in range(num_epochs):
        i = 0
        for degraded_imgs, mask_imgs, clean_imgs in data_loader:
            i+=1
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            denoised_img = model(degraded_imgs)
            loss = criterion(denoised_img, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if i%50 == 0:
                print(i, "/", len(data_loader))
        
        torch.save(model.state_dict(), os.path.join(weights_dir, f"unet_epoch_{epoch+1}.pth"))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")