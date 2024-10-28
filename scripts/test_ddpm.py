#!/usr/bin/env python3

import sys
# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '/home/hcl/bhanu/project')

import sys
import os
import pickle
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from models.ddpm import DDPM, UNet  # Ensure your models are correctly defined in models.ddpm
from psnr import calculate_psnr  # Ensure this function is defined and imported

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        for object_name in os.listdir(self.root_dir):
            object_path = os.path.join(self.root_dir, object_name, 'Val')
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
filename = "./dataloader_val_revisedv1.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    psnr_ls = []
    psnr_worst = []
    
    num_images = 0

    # Define paths
    test_weights_path = "../model_weights/ddpm_weights/ddpm_epoch_1.pth"  # Load final epoch weights

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)  # Move UNet model to device
    model.load_state_dict(torch.load(test_weights_path))
    model.eval()  # Set model to evaluation mode

    total_psnr = 0

    for i, (degraded_imgs, mask_imgs, clean_imgs) in enumerate(data_loader):
        degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)

        # Forward pass
        with torch.no_grad():
            denoised_img = model(degraded_imgs)

        # Ensure outputs and targets have the same shape
        # if denoised_img.shape != clean_imgs.shape:
        #     # Optionally resize denoised_img or clean_imgs here
        #     denoised_img = nn.functional.interpolate(denoised_img, size=clean_imgs.shape[2:], mode='bilinear', align_corners=True)

        # Calculate PSNR
        psnr_nlm = calculate_psnr(clean_imgs, denoised_img)
        worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)
        total_psnr += psnr_nlm
        num_images += 1

        psnr_ls.append(psnr_nlm)
        psnr_worst.append(worst_psnr)

        if num_images % 10 == 0:
            print(f'PSNR (UNET): {np.mean(psnr_ls):.2f} dB')
            print(f'PSNR (Degraded): {np.mean(psnr_worst):.2f} dB')

    # Calculate average PSNR
    avg_psnr = total_psnr / num_images if num_images > 0 else 0
    print(f"Average PSNR: {avg_psnr:.2f}")
