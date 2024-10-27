import numpy as np
import math
import cv2
import torch

# --------------------------------------------
# PSNR
# --------------------------------------------
def calculate_psnr(img1, img2, border=0):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1 (torch.Tensor): First image tensor with range [0, 255].
        img2 (torch.Tensor): Second image tensor with range [0, 255].
        border (int): Number of border pixels to exclude from the comparison.
        
    Returns:
        float: PSNR value. Returns infinity if images are identical.
    """
    # Convert tensors to numpy arrays
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    if img1.max() <= 1.0:
        img1 = img1 * 255.0
        img2 = img2 * 255.0

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    # Convert to float64 for precision in calculations
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)

    # Avoid division by zero
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR
    return 20 * math.log10(255.0 / math.sqrt(mse))

# --------------------------------------------
# SSIM
# --------------------------------------------
def calculate_ssim(img1, img2, border=0):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    #img1 = img1.squeeze()
    #img2 = img2.squeeze()
    
    img1 = img1.cpu().detach().numpy()
    img2 = img2.cpu().detach().numpy()

    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    h, w = img1.shape[:2]
    img1 = img1[border:h-border, border:w-border]
    img2 = img2[border:h-border, border:w-border]

    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1[:,:,i], img2[:,:,i]))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
