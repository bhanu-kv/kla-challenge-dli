# import numpy as np
# import math
# import cv2
# import torch

# # --------------------------------------------
# # PSNR
# # --------------------------------------------
# def calculate_psnr(img1, img2, border=0):
#     """
#     Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
#     Args:
#         img1 (torch.Tensor): First image tensor with range [0, 255].
#         img2 (torch.Tensor): Second image tensor with range [0, 255].
#         border (int): Number of border pixels to exclude from the comparison.
        
#     Returns:
#         float: PSNR value. Returns infinity if images are identical.
#     """
#     # Convert tensors to numpy arrays
#     img1 = img1.cpu().detach().numpy()
#     img2 = img2.cpu().detach().numpy()

#     if img1.max() <= 1.0:
#         img1 = img1 * 255.0
#         img2 = img2 * 255.0

#     if not img1.shape == img2.shape:
#         raise ValueError('Input images must have the same dimensions.')

#     h, w = img1.shape[:2]
#     img1 = img1[border:h-border, border:w-border]
#     img2 = img2[border:h-border, border:w-border]

#     # Convert to float64 for precision in calculations
#     img1 = img1.astype(np.float64)
#     img2 = img2.astype(np.float64)
#     mse = np.mean((img1 - img2) ** 2)

#     # Avoid division by zero
#     if mse == 0:
#         return float('inf')
    
#     # Calculate PSNR
#     return 20 * math.log10(255.0 / math.sqrt(mse))

# # --------------------------------------------

import numpy as np
import math
import torch
import cv2
# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1 (torch.Tensor or np.ndarray): First image tensor/array with range [0, 255].
        img2 (torch.Tensor or np.ndarray): Second image tensor/array with range [0, 255].
        border (int): Number of border pixels to exclude from the comparison.
        
    Returns:
        float: PSNR value. Returns infinity if images are identical.
    """
    # Convert to numpy if tensors
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy()

    # Check and adjust dynamic range
    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    # Ensure the input images are the same shape
    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    # Crop the border if specified
    if border > 0:
        img1 = img1[border:-border, border:-border, ...]
        img2 = img2[border:-border, border:-border, ...]

    # Convert to float for precision
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate MSE and PSNR
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * math.log10(255.0 / math.sqrt(mse))

################################################################################

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2, border=0):
    '''Calculate SSIM (Structural Similarity Index) between two images.'''
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    
    # Convert from PyTorch tensors to NumPy arrays if necessary
    if isinstance(img1, torch.Tensor):
        img1 = img1.cpu().detach().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.cpu().detach().numpy()
    
    # Ensure the images have the same shape
    if img1.shape != img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    
    # Exclude borders from the images
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    # Check the new dimensions
    # print("Cropped Image 1 shape:", img1.shape)
    # print("Cropped Image 2 shape:", img2.shape)

    # Calculate SSIM for multichannel images
    if img1.ndim == 3 and img1.shape[2] == 3:
        # Calculate SSIM across each channel and take the mean
        ssims = []
        for i in range(3):  # Iterate through the RGB channels
            channel_ssim = ssim(img1[..., i], img2[..., i], data_range=img1.max() - img1.min(), win_size=3)
            ssims.append(channel_ssim)
        return np.mean(ssims)  # Return the average SSIM across channels
    else:
        # For grayscale images or any other case
        return ssim(img1, img2, data_range=img1.max() - img1.min(), win_size=3)