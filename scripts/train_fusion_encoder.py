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
from models.fusion_encoder import FusionEncoder
from models.fft import fft_denoise
from models.bm3d import bm3d_batch_denoising
import cv2
from skimage.restoration import denoise_tv_chambolle
import torch
from models.model_nlm import nlm_patch_based_denoising_batch

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
    
    # Define paths
    test_weights_path = "../model_weights/unet_deep_weights/unet_epoch_30.pth"  # Load final epoch weights

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNetDenoiser().to(device)
    unet_model.load_state_dict(torch.load(test_weights_path))
    unet_model.eval()
    
    # Initialize model, optimizer, and loss function
    # test_weights_path = "../model_weights/fusion_weights/fusion_epoch_8.pth"  # Load final epoch weights
    fusion_model = FusionEncoder().to(device)
    # fusion_model.load_state_dict(torch.load(test_weights_path))
    
    optimizer = optim.Adam(fusion_model.parameters(), lr=0.0001, betas=(0.5, 0.999))
    criterion = nn.MSELoss()

    # Directory to save fusion_model weights
    weights_dir = "../model_weights/fusion_weights_new"
    os.makedirs(weights_dir, exist_ok=True)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        i = 0
        for degraded_imgs, mask_imgs, clean_imgs in data_loader:
            
            i+=1
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
            
            clean_imgs_fft = clean_imgs.detach().clone()
            clean_imgs_fft = clean_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
            
            degraded_imgs_fft = degraded_imgs.detach().clone()
            degraded_imgs_fft = degraded_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
            
            denoised_image_fft = fft_denoise(degraded_imgs_fft, low_pass=True, cutoff=40)
            
            # Forward pass
            with torch.no_grad():
                denoised_image_unet = unet_model(degraded_imgs)
            
            
            # # Move data to GPU
            denoised_image_fft = torch.from_numpy(denoised_image_fft).to(device).unsqueeze(0).unsqueeze(0)
            denoised_image_unet = denoised_image_unet.to(device)
            # denoised_image_bm3d = denoised_image_bm3d.to(device)
            
            ##################################################################################################################
            b, g, r = cv2.split(degraded_imgs.detach().cpu().numpy())[0][0]
            
            b_denoised = cv2.medianBlur(b, 3)
            g_denoised = cv2.medianBlur(g, 3)
            r_denoised = cv2.medianBlur(r, 3)
            
            denoised_image_median = cv2.merge([b_denoised, g_denoised, r_denoised])
            denoised_image_median = torch.from_numpy(denoised_image_median).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            ################################################################################################################
            hsv_image = cv2.cvtColor(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2HSV)
            denoised_image_bilateral = cv2.bilateralFilter(hsv_image, d=9, sigmaColor=75, sigmaSpace=75)

            # Convert back to BGR for display
            denoised_image_bilateral = cv2.cvtColor(denoised_image_bilateral, cv2.COLOR_HSV2BGR)
            denoised_image_bilateral = torch.from_numpy(denoised_image_bilateral).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            ################################################################################################################
            denoised_image_tv = denoise_tv_chambolle(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), weight=0.1)
            
            denoised_image_tv = torch.from_numpy(denoised_image_tv).unsqueeze(0).permute(0, 3, 1, 2).to(device)
            
            # Forward pass
            fused_imgs = fusion_model(denoised_image_fft, denoised_image_unet, denoised_image_median, denoised_image_bilateral, denoised_image_tv)
            
            # Compute loss
            loss = criterion(fused_imgs, clean_imgs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%50 == 0:
                print(i, "/", len(data_loader))
        
        torch.save(fusion_model.state_dict(), os.path.join(weights_dir, f"fusion_epoch_{epoch+1}.pth"))
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    