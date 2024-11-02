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
import numpy as np
from metrics.psnr import calculate_psnr
# from psnr_batch import calculate_psnr_batch
from display import display_images
from models.unet import UNetDenoiser
from metrics.psnr import calculate_ssim


class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        for object_name in os.listdir(self.root_dir):
            #print(object_name)
            object_path = os.path.join(self.root_dir, object_name, 'Val')
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
    
filename = "./dataloader_val_revisedv1.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

import os
from pathlib import Path
from torchvision.utils import save_image

val_dir = '../../Denoising_Dataset_train_val/'
results_dir = "../../Denoising_Dataset_results"

Path(results_dir).mkdir(parents=True, exist_ok=True)

def process_and_save_images(unet_model, val_dir, results_dir, transform):
    unet_model.eval()
     
    with torch.no_grad():
        for obj in os.listdir(val_dir):
            obj_dir = os.path.join(val_dir, obj, 'Val', 'Degraded_image')
            if not os.path.isdir(obj_dir):
                continue
            
            for defect_type in os.listdir(obj_dir):
                defect_dir = os.path.join(obj_dir, defect_type)
                save_defect_dir = os.path.join(results_dir, obj, 'Val', defect_type)
                Path(save_defect_dir).mkdir(parents=True, exist_ok=True)
                
                for img_name in os.listdir(defect_dir):
                    if img_name.endswith('.png'):
                        degraded_img_path = os.path.join(defect_dir, img_name)
                        mask_img_path = os.path.join(
                            val_dir, obj, 'Val', 'Defect_mask', defect_type, img_name.replace('.png', '_mask.png')
                        )

                        # Load and transform images
                        degraded_img = Image.open(degraded_img_path).convert('RGB')
                        mask_img = Image.open(mask_img_path).convert('L')
                        degraded_img = transform(degraded_img).to(device)
                        mask_img = transform(mask_img).to(device)

                        degraded_img = degraded_img.unsqueeze(0)
                        
                        # Forward pass through U-Net
                        denoised_image_unet = unet_model(degraded_img)
                        
                        B = denoised_image_unet.size(0)  # Number of images to display
                        
                        for i in range(B):
                            # Convert tensors to numpy arrays, scale to uint8, and convert to BGR
                            denoised_image = (denoised_image_unet[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            denoised_image_bgr = denoised_image[..., ::-1]  # Convert RGB to BGR
                            
                        # Save each deblurred image
                        output_path = os.path.join(save_defect_dir, f"{img_name.replace('.png', '')}.png")
                        Image.fromarray(denoised_image_bgr).save(output_path)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((900, 900))
])

if __name__ == '__main__':
    psnr_ls = []
    ssim_ls = []
    psnr_worst = []
    
    num_images = 0

    # Define paths
    test_weights_path = "../model_weights/unet_deep_weights/unet_epoch_48.pth"  # Load final epoch weights

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetDenoiser().to(device)
    model.load_state_dict(torch.load(test_weights_path))
    model.eval()

    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    print(len(data_loader))
    
    # for degraded_imgs, mask_imgs, clean_imgs in data_loader:
    #     degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
        
    #     # Forward pass
    #     with torch.no_grad():
    #         denoised_img = model(degraded_imgs)
        
    #     # Calculate PSNR
    #     psnr_nlm = calculate_psnr(clean_imgs, denoised_img)
    #     worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)
    #     ssim_nlm = calculate_ssim(clean_imgs, denoised_img)
        
    #     total_psnr += psnr_nlm
    #     total_ssim += ssim_nlm
    #     num_images += 1
        
    #     psnr_ls.append(psnr_nlm)
    #     ssim_ls.append(ssim_nlm)
    #     psnr_worst.append(worst_psnr)

    #     if num_images%10 == 0:
    #         print(f'PSNR: {np.mean(psnr_ls):.4f} dB')
    #         print(f'SSIM: {np.mean(ssim_ls):.4f}')

    #         # display_images(degraded_imgs, clean_imgs, denoised_img)
    #         # break
    # # Calculate average PSNR
    # avg_psnr = total_psnr / num_images
    # avg_ssim = total_ssim / num_images
    # print(f"Average PSNR: {avg_psnr:.2f}")
    # print(f"Average SSIM: {avg_ssim:.2f}")
    
    process_and_save_images(model, '../../Denoising_Dataset_train_val/', "../../Denoising_Dataset_results", transform)
