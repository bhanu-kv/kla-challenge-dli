#!/usr/bin/env python3

import torch
import cv2
import numpy as np

def nlm_denoise_image(image, mask, h=10, hForColor=10):
    """
    Apply NLM denoising directly on a single image, using the mask to preserve details.

    Args:
        image (torch.Tensor): The input image tensor (C, H, W) with values in [0, 1].
        mask (torch.Tensor): The mask tensor (H, W) where 0 indicates regions to be denoised.
        h (int): Parameter for controlling the filter strength for luminance.
        hForColor (int): Parameter for controlling the filter strength for color components.

    Returns:
        torch.Tensor: Denoised image with the same shape as the input image.
    """
    # Convert the image tensor to a NumPy array and scale to 0-255 for OpenCV
    img_np = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # Shape (H, W, C)
    
    # Apply NLM denoising on the full image
    denoised_np = cv2.fastNlMeansDenoisingColored(img_np, None, h, hForColor, 7, 21)
    
    # Convert back to tensor and normalize to [0, 1]
    denoised_tensor = torch.tensor(denoised_np).float().permute(2, 0, 1) / 255.0  # Shape (C, H, W)

    # Only update pixels outside mask (where mask is 0)
    mask_expanded = mask.expand_as(denoised_tensor)  # Ensure mask shape matches image channels
    denoised_tensor = torch.where(mask_expanded == 0, denoised_tensor, image)  # Combine using mask

    return denoised_tensor

def nlm_denoise_batch(images, masks, h=10, hForColor=10):
    """
    Apply NLM denoising directly on a batch of images with GPU support.

    Args:
        images (torch.Tensor): Batch of images (B, C, H, W) with values in [0, 1].
        masks (torch.Tensor): Batch of masks (B, H, W) with 0 indicating regions to be denoised.
        h (int): Parameter for controlling the filter strength for luminance.
        hForColor (int): Parameter for controlling the filter strength for color components.

    Returns:
        torch.Tensor: Batch of denoised images with the same shape as the input images.
    """
    denoised_images = []
    for image, mask in zip(images, masks):
        denoised_image = nlm_denoise_image(image, mask, h, hForColor)
        denoised_images.append(denoised_image)
    
    return torch.stack(denoised_images)  # Return as a tensor

# Example usage
# batch_images = torch.rand(4, 3, 256, 256).to('cuda')  # Example batch of images
# batch_masks = (torch.rand(4, 1, 256, 256) > 0.5).float().to('cuda')  # Example binary masks
# denoised_batch = nlm_denoise_batch(batch_images, batch_masks.squeeze(1), h=10, hForColor=10)
