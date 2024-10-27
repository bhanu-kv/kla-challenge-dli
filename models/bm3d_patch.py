import torch
import numpy as np
from bm3d import bm3d
from sklearn.feature_extraction import image

def extract_patches_parallel(img, patch_size=64, stride=64):
    """Extract patches in a parallel-friendly format for GPU processing."""
    img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert to (H, W, C)
    patches = image.extract_patches_2d(img_np, (patch_size, patch_size), max_patches=None, random_state=None)
    return torch.tensor(patches).float().permute(0, 3, 1, 2).to(img.device)  # Convert to (N, C, H, W)

def reconstruct_from_patches_parallel(patches, image_shape, patch_size=64, stride=64):
    """Reconstruct the image from patches with overlap averaging."""
    C, H, W = image_shape
    reconstructed_image = torch.zeros((C, H, W), dtype=torch.float32).to(patches.device)
    patch_count = torch.zeros((C, H, W), dtype=torch.float32).to(patches.device)

    patch_idx = 0
    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            reconstructed_image[:, i:i+patch_size, j:j+patch_size] += patches[patch_idx]
            patch_count[:, i:i+patch_size, j:j+patch_size] += 1
            patch_idx += 1

    reconstructed_image /= torch.maximum(patch_count, torch.ones_like(patch_count))
    return reconstructed_image

def bm3d_patch_based_denoising_parallel(image, sigma, patch_size=64, stride=64):
    """Apply BM3D denoising on patches in parallel using GPU."""
    patches = extract_patches_parallel(image, patch_size, stride)
    denoised_patches = []
    i = 0
    # Process all patches in parallel if possible
    for patch in patches:
        i+=1
        print(i, "/", len(patches))
        denoised_patch = bm3d(patch.cpu().permute(1, 2, 0).numpy(), sigma)  # (H, W, C)
        denoised_patch = torch.tensor(denoised_patch).float().permute(2, 0, 1).to(image.device)
        denoised_patches.append(denoised_patch)

    # Stack denoised patches and reconstruct the image
    denoised_patches = torch.stack(denoised_patches).to(image.device)
    return reconstruct_from_patches_parallel(denoised_patches, image.shape, patch_size, stride)

def bm3d_batch_denoising_parallel(images, sigma, patch_size=64, stride=64):
    """Apply parallelized BM3D patch-based denoising to a batch of images."""
    denoised_images = []
    for img in images:
        denoised_image = bm3d_patch_based_denoising_parallel(img, sigma, patch_size, stride)
        denoised_images.append(denoised_image)
    return torch.stack(denoised_images)
