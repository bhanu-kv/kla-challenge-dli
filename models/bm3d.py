#!/usr/bin/env python3

import torch
import numpy as np
from bm3d import bm3d

def bm3d_denoise(image, sigma):
    """
    Apply BM3D denoising to a single image.

    Args:
        image (numpy.ndarray): Input image as a NumPy array (H, W, C) in the [0, 255] range.
        sigma (float): Noise standard deviation.

    Returns:
        numpy.ndarray: Denoised image in the [0, 255] range.
    """

    # Scale image to [0, 255] if necessary
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    # Apply BM3D to denoise the image
    denoised_image = bm3d(image, sigma)
    
    # Scale back to [0, 255] if needed
    return denoised_image

def bm3d_batch_denoising(images, sigma):
    """
    Apply BM3D denoising to a batch of images.

    Args:
        images (torch.Tensor): Batch of images (B, C, H, W) in the [0, 1] range.
        sigma (float): Noise standard deviation.

    Returns:
        torch.Tensor: Batch of denoised images in the [0, 1] range.
    """
    denoised_images = []

    for img in images:
        # Convert image tensor to NumPy array and transpose to (H, W, C)
        img_np = img.permute(1, 2, 0).cpu().numpy()
        
        # Denoise the image
        denoised_np = bm3d_denoise(img_np, sigma)

        # Convert back to tensor and normalize to [0, 1]
        denoised_tensor = torch.tensor(denoised_np).float() / 255.0
        denoised_images.append(denoised_tensor.permute(2, 0, 1))  # Convert back to (C, H, W)

    return torch.stack(denoised_images).to(images.device)
