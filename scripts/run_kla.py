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
import numpy as np
from metrics.psnr import calculate_psnr
# from psnr_batch import calculate_psnr_batch
from display import display_images
from models.bm3d import bm3d_batch_denoising
from models.nlm_np_patch import nlm_denoise_batch
import cv2
from skimage.restoration import denoise_tv_chambolle
import torch
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

def save_generated_images(generated_images, output_dir, idx, prefix='generated'):
    """Save generated images to the specified output directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Convert the tensor image to a PIL image
    generated_images = generated_images.detach().cpu()  # Make sure to detach from the graph and move to CPU
    generated_images = transforms.ToPILImage()(generated_images)  # Assuming img is in C x H x W format
    
    # Construct filename
    img_filename = os.path.join(output_dir, f"{prefix}_img{idx}.png")
    
    # Save the image
    generated_images.save(img_filename)
        
filename = "./dataloader_train_revisedv1.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    os.makedirs('../../BM3D_Images/Train', exist_ok=True)
    psnr_ls = []
    psnr_worst = []
    ssim_ls = []
    i = 0
    
    total_psnr = 0
    total_ssim = 0
    num_images = 0

    for degraded_imgs, mask_imgs, clean_imgs in data_loader:
        i+=1
        print(i, "/", len(data_loader))
        ##################################################################################################################
        # noise_sigma = 45.0

        # # Perform BM3D denoising
        # denoised_image = bm3d_batch_denoising(degraded_imgs, noise_sigma)
        
        ##################################################################################################################
        denoised_image = nlm_denoise_batch(degraded_imgs, degraded_imgs.squeeze(1), h=10, hForColor=10)
        # denoised_image = denoised_image.squeeze(0)
        
        ##################################################################################################################
        # b, g, r = cv2.split(degraded_imgs.detach().cpu().numpy())[0][0]
        
        # b_denoised = cv2.medianBlur(b, 3)
        # g_denoised = cv2.medianBlur(g, 3)
        # r_denoised = cv2.medianBlur(r, 3)
        
        # denoised_image = cv2.merge([b_denoised, g_denoised, r_denoised])
        ################################################################################################################
        # hsv_image = cv2.cvtColor(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cv2.COLOR_BGR2HSV)
        # denoised_image = cv2.bilateralFilter(hsv_image, d=9, sigmaColor=75, sigmaSpace=75)

        # # Convert back to BGR for display
        # denoised_image = cv2.cvtColor(denoised_image, cv2.COLOR_HSV2BGR)
        
        ################################################################################################################
        # denoised_image = denoise_tv_chambolle(degraded_imgs.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), weight=0.1)
        
        # denoised_image = torch.from_numpy(denoised_image).unsqueeze(0).permute(0, 3, 1, 2)
        psnr_nlm = calculate_psnr(clean_imgs, denoised_image)
        worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)
        ssim_nlm = calculate_ssim(clean_imgs, denoised_image)
        
        total_psnr += psnr_nlm
        total_ssim += ssim_nlm
        num_images += 1
        
        psnr_ls.append(psnr_nlm)
        psnr_worst.append(worst_psnr)
        ssim_ls.append(ssim_nlm)
        
        
        # save_generated_images(denoised_image, output_dir = '../../BM3D_Images/Train', idx=i,prefix='bm3d')
        

        if i%1 == 0:
            print(f'PSNR: {np.mean(psnr_ls):.2f} dB')
            print(f'SSIM: {np.mean(ssim_ls):.4f}')
        
        # display_images(degraded_imgs, clean_imgs, denoised_image)
        # break
    
    print("Overall")
    avg_psnr = total_psnr / num_images
    avg_ssim = total_ssim / num_images
    print(f"Average PSNR: {avg_psnr:.2f}")
    print(f"Average SSIM: {avg_ssim:.2f}")