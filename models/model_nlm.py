#!/usr/bin/env python3

import numpy as np
import cv2
import torch
from sklearn.feature_extraction import image

def extract_patches(img, patch_size, stride):
    """Extract patches from the tensor image, handling batch dimension."""
    # Check if the input is a batch or a single image
    if img.dim() == 4:  # Batch of images (B, C, H, W)
        B, C, H, W = img.shape
        patches = []

        for b in range(B):
            img_np = img[b].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            img_patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None, random_state=None)
            patches.append(img_patches)

        return patches
    elif img.dim() == 3:  # Single image (C, H, W)
        C, H, W = img.shape
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        img_patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None, random_state=None)
        return [img_patches]  # Return as a list for consistency
    else:
        raise ValueError("Input tensor must have 3 (C, H, W) or 4 (B, C, H, W) dimensions.")

def reconstruct_image_from_patches(patches, original_shape, patch_size):
    """Reconstruct the tensor image from patches."""
    img_reconstructed = torch.zeros(original_shape, dtype=torch.float32)
    count_matrix = torch.zeros(original_shape, dtype=torch.float32)

    for (patch, i, j) in patches:
        img_reconstructed[:, i:i + patch_size, j:j + patch_size] += torch.tensor(patch).permute(0, 3, 1, 2)  # Convert back to (B, C, H, W)
        count_matrix[:, i:i + patch_size, j:j + patch_size] += 1

    img_reconstructed /= count_matrix
    return img_reconstructed

def nlm_denoise_patch(patch, mask_patch, h, hForColor):
    """Apply NLM on a single patch."""
    denoised_patch = patch.clone()
    patch_np = patch.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)

    nlm_denoised_full = cv2.fastNlMeansDenoisingColored(patch_np, None, h, hForColor, 7, 21)

    denoised_patch[mask_patch == 0] = torch.tensor(nlm_denoised_full)[mask_patch == 0]  # Convert back to tensor
    return denoised_patch

def nlm_patch_based_denoising_batch(images, masks, patch_size=64, stride=32, h=10, hForColor=10, max_patches=1000):
    """Apply patch-based NLM denoising to a batch of images."""
    denoised_images = []

    for image, mask in zip(images, masks):
        patches = extract_patches(image, patch_size, stride)
        mask_patches = extract_patches(mask, patch_size, stride)

        # Limit the number of patches processed
        patches = patches[0][:max_patches]  # Only take from the first image
        mask_patches = mask_patches[0][:max_patches]  # Only take from the first image

        denoised_patches = []
        for (patch, i, j), (mask_patch, _, _) in zip(patches, mask_patches):
            denoised_patch = nlm_denoise_patch(torch.tensor(patch).permute(2, 0, 1), torch.tensor(mask_patch).permute(2, 0, 1), h, hForColor)
            denoised_patches.append((denoised_patch, i, j))
        
        denoised_image = reconstruct_image_from_patches(denoised_patches, image.shape, patch_size)
        denoised_images.append(denoised_image)

    return torch.stack(denoised_images)  # Return as a tensor

# Example usage
# Assuming you have a batch of images and masks as tensors
# batch_images = torch.rand(4, 3, 256, 256)  # Batch of 4 images (B, C, H, W)
# batch_masks = (torch.rand(4, 1, 256, 256) > 0.5).float()  # Random binary masks as tensors

# Perform NLM denoising
# denoised_batch = nlm_patch_based_denoising_batch(batch_images, batch_masks.squeeze(1), patch_size=64, stride=32, h=10, hForColor=10)
