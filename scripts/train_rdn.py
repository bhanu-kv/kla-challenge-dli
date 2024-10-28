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
from models.rdn import ResidualDenseBlock
from models.rdn import RDN
from math import log10


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
    
filename = "./dataloader_train.pkl"
with open(filename, 'rb') as f:
    data_loader = pickle.load(f)

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RDN().to(device)  # Instantiate RDN model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    weights_dir = "../model_weights/rdn_weights"
    os.makedirs(weights_dir, exist_ok=True)

    num_epochs = 30
    for epoch in range(num_epochs):
        model.train()
        for i, (degraded_imgs, mask_imgs, clean_imgs) in enumerate(data_loader):
            degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
            denoised_img = model(degraded_imgs)
            loss = criterion(denoised_img, clean_imgs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print(f"Batch [{i}/{len(data_loader)}], Loss: {loss.item():.4f}")

        # Save model weights
        torch.save(model.state_dict(), os.path.join(weights_dir, f"rdn_epoch_{epoch+1}.pth"))
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

        # # Validation and PSNR calculation
        # model.eval()
        # with torch.no_grad():
        #     psnr_values = []
        #     for j, (degraded_imgs, mask_imgs, clean_imgs) in enumerate(data_loader):
        #         degraded_imgs, clean_imgs = degraded_imgs.to(device), clean_imgs.to(device)
        #         denoised_img = model(degraded_imgs)
        #         psnr = calculate_psnr(denoised_img, clean_imgs)
        #         psnr_values.append(psnr)

        #         # Save example denoised images
        #         if j < 5:  # Save a few examples per epoch
        #             for k in range(degraded_imgs.size(0)):
        #                 denoised_image_path = os.path.join(results_dir, f"epoch_{epoch+1}_img_{j*degraded_imgs.size(0)+k}.png")
        #                 transforms.ToPILImage()(denoised_img[k].cpu()).save(denoised_image_path)

        #     avg_psnr = sum(psnr_values) / len(psnr_values)
        #     print(f"Validation PSNR: {avg_psnr:.2f} dB")