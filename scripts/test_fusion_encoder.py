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
from metrics.psnr import calculate_ssim
from display import display_images
import cv2
from models.fft import fft_denoise
from models.bm3d import bm3d_batch_denoising
import cv2
from skimage.restoration import denoise_tv_chambolle
import torch
from models.model_nlm import nlm_patch_based_denoising_batch
from models.deblurmigan import Generator

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

val_dir = "F:/KLA Problem Statement/Denoising_Dataset_train_val"
results_dir = "F:/KLA Problem Statement/Denoising_Dataset_results"

Path(results_dir).mkdir(parents=True, exist_ok=True)

def process_and_save_images(unet_model, fusion_model, generator, val_dir, results_dir, transform):
    unet_model.eval()
    generator.eval()
    fusion_model.eval()
     
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
                        
                        degraded_imgs_fft = degraded_img.detach().clone()
                        degraded_imgs_fft = degraded_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
                        
                        denoised_image_fft = fft_denoise(degraded_imgs_fft, low_pass=True, cutoff=40)
    
                        # Forward pass
                        with torch.no_grad():
                            denoised_image_unet = unet_model(degraded_img)
                        
                        # # Move data to GPU
                        denoised_image_fft = torch.from_numpy(denoised_image_fft).to(device).unsqueeze(0).unsqueeze(0)
                        denoised_image_unet = denoised_image_unet.to(device)
                        
                        ##################################################################################################################
                        b, g, r = cv2.split(degraded_img.detach().cpu().numpy())[0][0]
                        
                        b_denoised = cv2.medianBlur(b, 3)
                        g_denoised = cv2.medianBlur(g, 3)
                        r_denoised = cv2.medianBlur(r, 3)
                        
                        denoised_image_median = cv2.merge([b_denoised, g_denoised, r_denoised])
                        denoised_image_median = torch.from_numpy(denoised_image_median).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                        ################################################################################################################
                        hsv_image = cv2.cvtColor(degraded_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2HSV)
                        denoised_image_bilateral = cv2.bilateralFilter(hsv_image, d=9, sigmaColor=75, sigmaSpace=75)

                        # Convert back to BGR for display
                        denoised_image_bilateral = cv2.cvtColor(denoised_image_bilateral, cv2.COLOR_HSV2BGR)
                        denoised_image_bilateral = torch.from_numpy(denoised_image_bilateral).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                        ################################################################################################################
                        denoised_image_tv = denoise_tv_chambolle(degraded_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), weight=0.1)
                        
                        denoised_image_tv = torch.from_numpy(denoised_image_tv).unsqueeze(0).permute(0, 3, 1, 2).to(device)
                        
                        # display_images(degraded_imgs, clean_imgs, denoised_image_unet)
                        # break
                        
                        # Forward pass
                        with torch.no_grad():
                            fused_imgs = fusion_model(denoised_image_fft, denoised_image_unet, denoised_image_median, denoised_image_bilateral, denoised_image_tv)
                        
                        with torch.no_grad():
                            output = generator(fused_imgs)
                        
                        B = output.size(0)  # Number of images to display
                        
                        for i in range(B):
                            # Convert tensors to numpy arrays, scale to uint8, and convert to BGR
                            denoised_image = (output[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
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
    psnr_worst = []
    
    ssim_ls = []
    ssim_worst = []
    
    # Define paths
    test_weights_path = "../model_weights/unet_deep_weights/unet_epoch_48.pth"  # Load final epoch weights

    # Load model and weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unet_model = UNetDenoiser().to(device)
    unet_model.load_state_dict(torch.load(test_weights_path))
    unet_model.eval()
    
    # Initialize model, optimizer, and loss function
    test_weights_path = "../model_weights/fusion_weights_new/fusion_epoch_20.pth"  # Load final epoch weights
    fusion_model = FusionEncoder().to(device)
    fusion_model.load_state_dict(torch.load(test_weights_path))
    fusion_model.eval()
    
    test_weights_path = "../model_weights/migan/generator_epoch_99.pth"  # Load final epoch weights
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(test_weights_path))
    generator.eval()
    
    total_psnr = 0
    total_ssim = 0
    num_images = 0
    
    # for degraded_imgs, mask_imgs, clean_imgs in data_loader:
    #     degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
        
    #     clean_imgs_fft = clean_imgs.detach().clone()
    #     clean_imgs_fft = clean_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
        
    #     degraded_imgs_fft = degraded_imgs.detach().clone()
    #     degraded_imgs_fft = degraded_imgs_fft[0].permute(1, 2, 0).cpu().numpy()
        
    #     denoised_image_fft = fft_denoise(degraded_imgs_fft, low_pass=True, cutoff=40)
        
    #     # Display the image
    #     # cv2.imshow('Grayscale Image', denoised_image_fft)
    #     # cv2.waitKey(0)  # Wait for a key press to close the window
    #     # cv2.destroyAllWindows()

    #     # Forward pass
    #     with torch.no_grad():
    #         denoised_image_unet = unet_model(degraded_imgs)
        
        
    #     # # Move data to GPU
    #     denoised_image_fft = torch.from_numpy(denoised_image_fft).to(device).unsqueeze(0).unsqueeze(0)
    #     denoised_image_unet = denoised_image_unet.to(device)
        
    #     ##################################################################################################################
    #     b, g, r = cv2.split(degraded_imgs.detach().cpu().numpy())[0][0]
        
    #     b_denoised = cv2.medianBlur(b, 3)
    #     g_denoised = cv2.medianBlur(g, 3)
    #     r_denoised = cv2.medianBlur(r, 3)
        
    #     denoised_image_median = cv2.merge([b_denoised, g_denoised, r_denoised])
    #     denoised_image_median = torch.from_numpy(denoised_image_median).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    #     ################################################################################################################
    #     hsv_image = cv2.cvtColor(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2HSV)
    #     denoised_image_bilateral = cv2.bilateralFilter(hsv_image, d=9, sigmaColor=75, sigmaSpace=75)

    #     # Convert back to BGR for display
    #     denoised_image_bilateral = cv2.cvtColor(denoised_image_bilateral, cv2.COLOR_HSV2BGR)
    #     denoised_image_bilateral = torch.from_numpy(denoised_image_bilateral).unsqueeze(0).permute(0, 3, 1, 2).to(device)
    #     ################################################################################################################
    #     denoised_image_tv = denoise_tv_chambolle(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), weight=0.1)
        
    #     denoised_image_tv = torch.from_numpy(denoised_image_tv).unsqueeze(0).permute(0, 3, 1, 2).to(device)
        
    #     # display_images(degraded_imgs, clean_imgs, denoised_image_unet)
    #     # break
        
    #     # Forward pass
    #     with torch.no_grad():
    #         fused_imgs = fusion_model(denoised_image_fft, denoised_image_unet, denoised_image_median, denoised_image_bilateral, denoised_image_tv)
        
    #     with torch.no_grad():
    #         fused_imgs = generator(fused_imgs)
                            
    #     # Calculate PSNR
    #     psnr_nlm = calculate_psnr(clean_imgs, fused_imgs)
    #     worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)
        
    #     ssim_nlm = calculate_ssim(clean_imgs, fused_imgs)
    #     # worst_ssim = calculate_ssim(clean_imgs, degraded_imgs)
        
    #     total_psnr += psnr_nlm
    #     total_ssim += ssim_nlm
    #     num_images += 1
        
    #     psnr_ls.append(psnr_nlm)
    #     psnr_worst.append(worst_psnr)
    #     ssim_ls.append(ssim_nlm)
    #     # ssim_worst.append(worst_ssim)

    #     if num_images%1 == 0:
    #         print(num_images)
    #         print(f'PSNR (Fused): {np.mean(psnr_ls):.4f} dB')
    #         # print(f'PSNR (Degraded): {np.mean(psnr_worst):.4f} dB')
            
    #         print(f'SSIM (Fused): {np.mean(ssim_ls):.4f}')
    #         # print(f'SSIM (Degraded): {np.mean(ssim_worst):.4f}')

    #         # display_images(degraded_imgs, clean_imgs, fused_imgs)
    #         # break
    # # Calculate average PSNR
    # avg_psnr = total_psnr / num_images
    # avg_ssim = total_ssim / num_images
    # print(f"Average PSNR: {avg_psnr:.2f}")
    # print(f"Average SSIM: {avg_ssim:.2f}")
    
    process_and_save_images(unet_model, fusion_model, generator, '../../Denoising_Dataset_train_val/', "../../Denoising_Dataset_results", transform)
    