import numpy as np
import torch

def calculate_psnr_batch(original_images, denoised_images):
    """
    Calculate PSNR for a batch of images.
    
    Args:
        original_images (torch.Tensor): Batch of original clean images (B, C, H, W).
        denoised_images (torch.Tensor): Batch of denoised images (B, C, H, W).
        
    Returns:
        float: Average PSNR for the batch.
    """
    psnr_values = []

    # Iterate through each pair of images in the batch
    for i in range(original_images.size(0)):
        original_np = original_images[i].permute(1, 2, 0).cpu().numpy()
        denoised_np = denoised_images[i].permute(1, 2, 0).cpu().numpy()

        # Ensure images are in the right format (uint8)
        original_np = (original_np * 255).astype(np.uint8)
        denoised_np = (denoised_np * 255).astype(np.uint8)

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((original_np - denoised_np) ** 2)

        # To avoid division by zero
        if mse == 0:
            psnr_values.append(float('inf'))  # PSNR is infinite if there is no error
        else:
            max_pixel = 255.0  # Maximum possible pixel value (for 8-bit images)
            psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
            psnr_values.append(psnr)

    # Return the average PSNR for the batch
    return np.mean(psnr_values)