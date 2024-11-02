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
import numpy as np
from metrics.psnr import calculate_psnr
from display import display_images
from models.deblurmigan import Generator
from models.deblurmigan import Discriminator
import cv2

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

# Custom Mask-Aware Loss
def mask_aware_loss(fake_img, real_img, mask, denoised_img):
    # L1 loss inside the mask (defect region)
    l1loss = nn.L1Loss()
    loss_in_mask = l1loss(fake_img * mask, real_img * mask)
    # L2 loss across the entire image (including outside mask)
    l2loss = nn.MSELoss()
    loss_outside_mask = l2loss(fake_img, real_img)
    loss_prev_image = l2loss(fake_img, denoised_img)
    return (100*loss_in_mask + 100*loss_outside_mask + 2000*loss_prev_image)

if __name__ == '__main__':
    # Define paths
    test_weights_path = "../model_weights/unet_deep_weights/unet_epoch_48.pth"  # Load final epoch weights

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNetDenoiser().to(device)
    unet_model.load_state_dict(torch.load(test_weights_path))
    unet_model.eval()
    
    # Initialize model, optimizer, and loss function
    # test_weights_path = "../model_weights/fusion_weights/fusion_epoch_10.pth"  # Load final epoch weights
    # fusion_model = FusionEncoder().to(device)
    # fusion_model.load_state_dict(torch.load(test_weights_path))
    # fusion_model.eval()
    
    # Instantiate models and move to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_weights_path = "../model_weights/migan_weights_new_ka_new/generator_newer_newer_best.pth"
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(test_weights_path))
    discriminator = Discriminator().to(device)

    criterion_gan = nn.BCELoss()
    # criterion_content = nn.L1Loss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999))
    
    epochs = 300
    best_score = 0
    
    # Directory to save fusion_model weights
    weights_dir = "../model_weights/migan_weights_new_ka_new"
    os.makedirs(weights_dir, exist_ok=True)

    for epoch in range(epochs):
        i = 0
        epoch_score = 0
        
        test_weights_path = "../model_weights/migan_weights_new_ka_new/generator_newer_newer_best.pth"
        generator.load_state_dict(torch.load(test_weights_path))
        
        for degraded_imgs, mask_imgs, clean_imgs in data_loader:
            i+=1
            
            batch_size = degraded_imgs.size(0)
            degraded_imgs, mask_imgs, clean_imgs = degraded_imgs.to(device), mask_imgs.to(device), clean_imgs.to(device)
            
            # clean_imgs_fft = clean_imgs.detach().clone()
            # clean_imgs_fft = clean_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
            
            # degraded_imgs_fft = degraded_imgs.detach().clone()
            # degraded_imgs_fft = degraded_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
            
            # denoised_image_fft = fft_denoise(degraded_imgs_fft, low_pass=True, cutoff=40)

            # Forward pass
            with torch.no_grad():
                denoised_image_unet = unet_model(degraded_imgs)
            
            
            # # Move data to GPU
            # denoised_image_fft = torch.from_numpy(denoised_image_fft).to(device).unsqueeze(0).unsqueeze(0)
            denoised_image_unet = denoised_image_unet.to(device)
            
            # Forward pass
            # with torch.no_grad():
                # fused_imgs = fusion_model(denoised_image_fft, denoised_image_unet)
                
            # Labels for GAN
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)
            
            ### Train Discriminator
            optimizer_d.zero_grad()
            outputs = discriminator(clean_imgs)
            
            d_loss_real = criterion_gan(outputs, real_labels)

            fake_imgs = generator(denoised_image_unet)
            outputs = discriminator(fake_imgs.detach())
            d_loss_fake = criterion_gan(outputs, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            optimizer_d.step()

            ### Train Generator
            optimizer_g.zero_grad()
            outputs = discriminator(fake_imgs)
            g_loss_gan = criterion_gan(outputs, real_labels)/100

            # Content loss using L1 loss
            # mask_weighted_content_loss = criterion_content(fake_imgs * mask_imgs, clean_imgs * mask_imgs)

            # Combine losses
            g_mask_aware_loss = mask_aware_loss(fake_imgs, clean_imgs, mask_imgs, denoised_image_unet)  # Mask-aware loss
            total_g_loss = g_loss_gan + g_mask_aware_loss
            # print(g_loss_gan, g_mask_aware_loss)
            total_g_loss.backward()
            optimizer_g.step()
            epoch_score += total_g_loss.item()
            if i%50 == 0:
                print(g_mask_aware_loss.item(), g_loss_gan.item())
                print(i, "/", len(data_loader))
        
        if epoch_score > best_score:
            best_score = epoch_score
            torch.save(generator.state_dict(), os.path.join(weights_dir, f"generator_newer_newer_best.pth"))
        
        print(f"Epoch [{epoch + 1}/{epochs}], D Loss: {d_loss.item()}, G Loss: {total_g_loss.item()}")
        
    