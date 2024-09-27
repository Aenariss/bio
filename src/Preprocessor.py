"""
BIO Project 2024

This module contains a class to preprocess an image 
Author: Filip Brna <xbrnaf00> + ChatGPT
"""

import numpy as np
import cv2
from scipy import ndimage

class Preprocessor:
    def __init__(self):
        pass

    def roi(self, image):
        # Define the region of interest (ROI) in the image
        height, width = image.shape
        roi = np.zeros_like(image, dtype=bool)
        roi[height // 4:3 * height // 4, width // 4:3 * width // 4] = True
        return roi

    def detect_finger_vein_mask(self, image):
        # Apply threshold to separate finger from background
        thresh = self.roi(image)
        binary = np.zeros_like(image, dtype=bool)
        binary[image > thresh] = True

        # Remove unwanted regions where the finger is not present
        binary = binary.astype(np.uint8) * 255
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # Fill holes in the binary mask
        filled_mask = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

        return filled_mask

    def preprocess_image(self, image):
        # Generate the vein mask
        vein_mask = self.detect_finger_vein_mask(image)
        # Apply the mask to isolate the finger region
        masked_image = cv2.bitwise_and(image, image, mask=vein_mask)
        # Contrast Limited Adaptive Histogram Equalization (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(masked_image)
        # Gaussian blur to reduce noise
        blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
        # Unsharp masking to enhance details
        sharp_image = cv2.addWeighted(clahe_image, 1.5, blurred_image, -0.5, 0)

        return vein_mask, masked_image, clahe_image, blurred_image, sharp_image
