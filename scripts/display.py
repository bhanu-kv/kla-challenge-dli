#!/usr/bin/env python3

import cv2
import numpy as np

def display_images(degraded, clean, denoised):
    """Display degraded, clean, and denoised images using OpenCV."""
    B = degraded.size(0)  # Number of images to display

    for i in range(B):
        # Convert tensors to numpy arrays and scale to uint8
        degraded_image = (degraded[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        clean_image = (clean[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        denoised_image = (denoised[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        # Display the images in separate windows
        cv2.imshow(f'Degraded Image {i+1}', degraded_image)
        cv2.imshow(f'Clean Image {i+1}', clean_image)
        cv2.imshow(f'Denoised Image (NLM) {i+1}', denoised_image)

    cv2.waitKey(0)  # Wait for a key press to close the windows
    cv2.destroyAllWindows()  # Close all OpenCV windows