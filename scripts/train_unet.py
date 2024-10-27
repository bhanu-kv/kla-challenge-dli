#!/usr/bin/env python3

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/bhanu/IITM/Courses/DLI/KLA/project')

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
import numpy as np
from models.model_nlm_gpu import nlm_patch_based_denoising_batch_gpu
from psnr import calculate_psnr
# from psnr_batch import calculate_psnr_batch
from display import display_images
from models.bm3d import bm3d_batch_denoising
from models.nlm_np_patch import nlm_denoise_batch
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
    
filename = "./dataloader_val_revisedv3.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    psnr_ls = []
    psnr_worst = []
    
    i = 0
    
    # Assuming UNetDenoiser and train_loader are already defined
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoiser().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Directory to save model weights
    weights_dir = "../model_weights/unet_weights"
    os.makedirs(weights_dir, exist_ok=True)

    # Training loop
    num_epochs = 20

    for epoch in range(num_epochs):
        for degraded_imgs, mask_imgs, clean_imgs in data_loader:
            # print(degraded_imgs.shape)
            # print(mask_imgs.shape)
            # print(clean_imgs.shape)
            i+=1
            print(i, "/", len(data_loader))
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
            
            # Forward pass
            denoised_img = model(degraded_imgs)
            loss = criterion(denoised_img, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        torch.save(model.state_dict(), os.path.join(weights_dir, f"unet_epoch_{epoch+1}.pth"))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

            # psnr_nlm = calculate_psnr(clean_imgs, denoised_img)
            # worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)

            # psnr_ls.append(psnr_nlm)
            # psnr_worst.append(worst_psnr)

            # if i%10 == 0:
            #     print(f'PSNR (NLM): {np.mean(psnr_ls):.2f} dB')
            #     print(f'PSNR (WORST): {np.mean(psnr_worst):.2f} dB')
            
            # display_images(degraded_imgs, clean_imgs, denoised_img)
            # break
        
        # print("Overall")
        # print(f'PSNR (NLM): {np.mean(psnr_nlm):.2f} dB')
        # print(f'PSNR (WORST): {np.mean(worst_psnr):.2f} dB')

        # display_images(degraded_imgs, clean_imgs, denoised_image)