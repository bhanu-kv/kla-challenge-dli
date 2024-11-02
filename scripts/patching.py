#!/usr/bin/env python3

import numpy as np

def extract_patches(image, patch_size=64, stride=32):
    patches = []
    h, w = image.shape[:2]
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append((patch, i, j))
    return patches

def reconstruct_image_from_patches(patches, image_shape, patch_size=64, stride=32):
    reconstructed_image = np.zeros(image_shape, dtype=np.float32)
    weight_map = np.zeros(image_shape, dtype=np.float32)
    
    for patch, i, j in patches:
        reconstructed_image[i:i+patch_size, j:j+patch_size] += patch
        weight_map[i:i+patch_size, j:j+patch_size] += 1
    
    # Avoid division by zero and normalize
    reconstructed_image /= np.maximum(weight_map, 1)
    return reconstructed_image.astype(np.uint8)