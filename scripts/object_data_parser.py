#!/usr/bin/env python3

import sys
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pickle

class DefectDataset(Dataset):
    def __init__(self, root_dir, transform=None, object_type=None):
        self.root_dir = root_dir
        self.transform = transform
        self.object_type = object_type
        self.triplets = self._load_triplets()

    def _load_triplets(self):
        triplets = []
        object_path = os.path.join(self.root_dir, self.object_type, 'Val')
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

                # Only add the triplet if all three files exist
                if os.path.exists(degraded_img) and os.path.exists(mask_img) and os.path.exists(clean_img):
                    triplets.append((degraded_img, mask_img, clean_img))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        degraded_img_path, mask_img_path, clean_img_path = self.triplets[idx]

        try:
            degraded_img = Image.open(degraded_img_path).convert("RGB")
            mask_img = Image.open(mask_img_path).convert("L")  # Convert mask to grayscale
            clean_img = Image.open(clean_img_path).convert("RGB")
        except IOError as e:
            print(f"Error loading images: {e}")
            return None  # or handle the error as appropriate

        # Apply transforms, if any
        if self.transform:
            degraded_img = self.transform(degraded_img)
            mask_img = self.transform(mask_img)
            clean_img = self.transform(clean_img)

        return degraded_img, mask_img, clean_img

transform = transforms.Compose([
    transforms.Resize((900, 900)),  # Change the size according to the model
    transforms.ToTensor(),
])

if __name__ == '__main__':
    root_dir = '../../Denoising_Dataset_train_val/'
    object_types = os.listdir(root_dir)  # List all object types in the root directory

    for object_type in object_types:
        # Create dataset and DataLoader for each object type
        dataset = DefectDataset(root_dir=root_dir, transform=transform, object_type=object_type)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)  # Consider increasing num_workers
        
        # Save DataLoader to a separate pickle file
        with open(f'object_{object_type}.pkl', 'wb') as f:
            pickle.dump(data_loader, f)
