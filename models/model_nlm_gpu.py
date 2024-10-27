#!/usr/bin/env python3

import torch
import numpy as np
import cv2
from sklearn.feature_extraction import image

def extract_patches(img, patch_size, stride):
    """Extract patches from the tensor image, handling batch dimension."""
    if img.dim() == 4:  # Batch of images (B, C, H, W)
        B, C, H, W = img.shape
        patches = []

        for b in range(B):
            img_np = img[b].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
            img_patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None, random_state=None)

            # Save index information
            for i in range(0, H - patch_size + 1, stride):
                for j in range(0, W - patch_size + 1, stride):
                    patches.append((img_patches[(i // stride) * ((W - patch_size) // stride + 1) + (j // stride)], i, j))

        return patches
    elif img.dim() == 3:  # Single image (C, H, W)
        C, H, W = img.shape
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
        img_patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None, random_state=None)

        return [(img_patches[i], i // stride, i % stride) for i in range(len(img_patches))]  # Index information
    else:
        raise ValueError("Input tensor must have 3 (C, H, W) or 4 (B, C, H, W) dimensions.")

def reconstruct_image_from_patches(patches, original_shape, patch_size):
    """Reconstruct the tensor image from patches."""
    img_reconstructed = torch.zeros(original_shape, dtype=torch.float32).to('cuda')
    count_matrix = torch.zeros(original_shape, dtype=torch.float32).to('cuda')

    for (patch, i, j) in patches:
        img_reconstructed[:, i:i + patch_size, j:j + patch_size] += patch.permute(2, 0, 1)  # Convert back to (C, H, W)
        count_matrix[:, i:i + patch_size, j:j + patch_size] += 1

    img_reconstructed /= count_matrix
    img_reconstructed = torch.clamp(img_reconstructed, 0, 1)  # Clamp values to be in [0, 1]
    return img_reconstructed

def nlm_denoise_patch_gpu(patch, mask_patch, h, hForColor):
    """Apply NLM on a single patch using GPU."""
    # Move patch and mask to GPU
    patch = patch.to('cuda')
    mask_patch = mask_patch.to('cuda')  # Ensure mask is also on GPU
    denoised_patch = patch.clone()

    # Convert patch from tensor to numpy and scale to 0-255 for uint8
    patch_np = patch.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
    patch_np = (patch_np * 255).astype(np.uint8)  # Scale to uint8

    # Apply NLM using OpenCV
    nlm_denoised_full = cv2.fastNlMeansDenoisingColored(patch_np, None, h, hForColor, 7, 21)

    # Convert back to tensor and normalize to 0-1 range
    nlm_denoised_full_tensor = torch.tensor(nlm_denoised_full).to('cuda').float() / 255.0  # Shape: [H, W, C]

    # Adjust mask shape to match the denoised tensor for indexing
    if mask_patch.dim() == 3:  # If mask is [1, H, W]
        mask_patch = mask_patch.squeeze(0)  # Shape: [H, W]

    # Ensure the mask is boolean
    mask_boolean = mask_patch == 0  # Create a boolean mask for areas without defects

    # Combine original and denoised based on mask
    # Change denoised_patch to [H, W, C] for proper indexing

    denoised_patch = denoised_patch.permute(1, 2, 0)  # Convert to (H, W, C)
    mask_boolean = mask_boolean.permute(1, 2, 0)

    # Update denoised_patch where mask_boolean is True
    # Ensure correct shapes: 
    # denoised_patch [H, W, C] and nlm_denoised_full_tensor [H, W, C]
    denoised_patch[mask_boolean] = nlm_denoised_full_tensor[mask_boolean]

    return denoised_patch  # Return back to [C, H, W]



def nlm_patch_based_denoising_batch_gpu(images, masks, patch_size=128, stride=64, h=10, hForColor=10, max_patches=100):
    """Apply patch-based NLM denoising to a batch of images using GPU."""
    denoised_images = []

    for image, mask in zip(images, masks):
        patches = extract_patches(image, patch_size, stride)
        mask_patches = extract_patches(mask, patch_size, stride)

        # Limit the number of patches processed
        patches = patches[:max_patches]  # Only take the first max_patches
        mask_patches = mask_patches[:max_patches]  # Only take the first max_patches

        denoised_patches = []
        for (patch, i, j), (mask_patch, _, _) in zip(patches, mask_patches):
            # Convert numpy patches back to tensors
            patch_tensor = torch.tensor(patch).permute(2, 0, 1).float()  # (C, H, W)
            mask_tensor = torch.tensor(mask_patch).permute(2, 0, 1).float()  # (C, H, W)

            denoised_patch = nlm_denoise_patch_gpu(patch_tensor, mask_tensor, h, hForColor)
            denoised_patches.append((denoised_patch, i, j))
        
        denoised_image = reconstruct_image_from_patches(denoised_patches, image.shape, patch_size)
        denoised_images.append(denoised_image)

    return torch.stack(denoised_images).to('cuda')  # Return as a tensor

# # Example usage
# # Assuming you have a batch of images and masks as tensors on GPU
# # batch_images = torch.rand(4, 3, 256, 256).to('cuda')  # Batch of 4 images (B, C, H, W)
# # batch_masks = (torch.rand(4, 1, 256, 256) > 0.5).float().to('cuda')  # Random binary masks as tensors

# # Perform NLM denoising
# # denoised_batch = nlm_patch_based_denoising_batch_gpu(batch_images, batch_masks.squeeze(1), patch_size=64, stride=32, h=10, hForColor=10)

# import torch
# import cv2
# import numpy as np
# from sklearn.feature_extraction import image

# def extract_patches(img, patch_size, stride):
#     """Extract patches from tensor image in batch mode, retaining GPU compatibility."""
#     B, C, H, W = img.shape
#     patches = []

#     for b in range(B):
#         img_np = img[b].permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C) for extraction
#         img_patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None)
        
#         # Collect patches and indices
#         for i in range(0, H - patch_size + 1, stride):
#             for j in range(0, W - patch_size + 1, stride):
#                 idx = (i // stride) * ((W - patch_size) // stride + 1) + (j // stride)
#                 patches.append((img_patches[idx], i, j))

#     return patches

# def reconstruct_image_from_patches(patches, original_shape, patch_size):
#     """Reconstruct image from patches, averaging overlaps."""
#     img_reconstructed = torch.zeros(original_shape, dtype=torch.float32).to('cuda')
#     count_matrix = torch.zeros(original_shape, dtype=torch.float32).to('cuda')

#     for (patch, i, j) in patches:
#         img_reconstructed[:, i:i + patch_size, j:j + patch_size] += patch.permute(2, 0, 1)
#         count_matrix[:, i:i + patch_size, j:j + patch_size] += 1

#     img_reconstructed /= torch.maximum(count_matrix, torch.ones_like(count_matrix))
#     return torch.clamp(img_reconstructed, 0, 1)  # Ensure values are in [0, 1]

# def nlm_denoise_patch_gpu(patch, mask_patch, h, hForColor):
#     """Apply NLM on a single patch using GPU-friendly tensor operations."""
#     patch_np = (patch.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
#     nlm_denoised_full = cv2.fastNlMeansDenoisingColored(patch_np, None, h, hForColor, 7, 21)
#     nlm_denoised_full_tensor = torch.tensor(nlm_denoised_full).to('cuda').float() / 255.0  # Scale to [0, 1]

#     # Apply mask to combine original and denoised content where needed
#     denoised_patch = patch.clone()
#     mask_boolean = mask_patch == 0
#     denoised_patch[mask_boolean] = nlm_denoised_full_tensor.permute(2, 0, 1)[mask_boolean]
    
#     return denoised_patch  # Return in (C, H, W) format

# def nlm_patch_based_denoising_batch_gpu(images, masks, patch_size=64, stride=32, h=10, hForColor=10, max_patches=1000):
#     """Denoise batch of images using patch-based NLM with GPU support."""
#     denoised_images = []
#     for image, mask in zip(images, masks):
#         patches = extract_patches(image, patch_size, stride)
#         mask_patches = extract_patches(mask, patch_size, stride)
        
#         # Only take first max_patches for efficiency
#         patches = patches[:max_patches]
#         mask_patches = mask_patches[:max_patches]

#         denoised_patches = []
#         for (patch, i, j), (mask_patch, _, _) in zip(patches, mask_patches):
#             patch_tensor = torch.tensor(patch).permute(2, 0, 1).float().to('cuda') / 255.0
#             mask_tensor = torch.tensor(mask_patch).float().to('cuda')
            
#             denoised_patch = nlm_denoise_patch_gpu(patch_tensor, mask_tensor, h, hForColor)
#             denoised_patches.append((denoised_patch, i, j))
        
#         denoised_image = reconstruct_image_from_patches(denoised_patches, image.shape, patch_size)
#         denoised_images.append(denoised_image)

#     return torch.stack(denoised_images)
