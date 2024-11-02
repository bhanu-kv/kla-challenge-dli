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
import cv2
import numpy as np
from psnr import calculate_psnr
from display import display_images


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
    
filename = "./dataloader_val_revisedv3.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

def fft_denoise(image, low_pass=True, cutoff=25):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask for low-pass filtering
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)

    if low_pass:
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    else:
        mask = np.ones((rows, cols), dtype=np.uint8)
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # Apply the mask
    fshift_filtered = fshift * mask

    # Inverse FFT to reconstruct the image
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

if __name__ == '__main__':
    psnr_ls = []
    psnr_worst = []
    i = 0

    for degraded_imgs, mask_imgs, clean_imgs in data_loader:
        # i+=1
        print(i, "/", len(data_loader))

        clean_imgs = clean_imgs[0]
        clean_imgs = clean_imgs.permute(1, 2, 0)
        clean_imgs = clean_imgs.detach().cpu().numpy()
        
        degraded_imgs = degraded_imgs[0]
        degraded_imgs = degraded_imgs.permute(1, 2, 0)
        degraded_imgs = degraded_imgs.detach().cpu().numpy()
        
        denoised_image = fft_denoise(degraded_imgs, low_pass=True, cutoff=40)
        
        clean_imgs = cv2.cvtColor(clean_imgs, cv2.COLOR_BGR2GRAY)
        degraded_imgs = cv2.cvtColor(degraded_imgs, cv2.COLOR_BGR2GRAY)
        
        psnr_nlm = calculate_psnr(clean_imgs, denoised_image)
        worst_psnr = calculate_psnr(clean_imgs, degraded_imgs)

        psnr_ls.append(psnr_nlm)
        psnr_worst.append(worst_psnr)

        if i%10 == 0:
            print(f'PSNR (FFT): {np.mean(psnr_ls):.2f} dB')
            print(f'PSNR (WORST): {np.mean(psnr_worst):.2f} dB')
        
        # # Display the denoised image
        # cv2.imshow(f'Degraded Image', degraded_imgs)
        # cv2.imshow(f'Clean Image', clean_imgs)
        # cv2.imshow(f'Denoised Image (FFT)', denoised_image)
        # cv2.waitKey(0)  # Wait for a key press to close the window
        # cv2.destroyAllWindows()  # Close the window
        # break
    
    print("Overall")
    print(f'PSNR (FFT): {np.mean(psnr_ls):.2f} dB')
    print(f'PSNR (WORST): {np.mean(psnr_worst):.2f} dB')

    # display_images(degraded_imgs, clean_imgs, denoised_image)