import cv2
import numpy as np

def fft_denoise(image, low_pass=True, cutoff=25):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply FFT
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask for low-pass filtering
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.uint8)

    if low_pass:
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 1
    else:
        mask = np.ones((rows, cols), dtype=np.uint8)
        mask[crow - cutoff:crow + cutoff, ccol - cutoff:ccol + cutoff] = 0

    # Apply the mask
    fshift_filtered = fshift * mask

    # Inverse FFT to reconstruct the image
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back