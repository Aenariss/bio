"""
BIO Project 2024

This module contains a class to preprocess an image 
Author: 
    Filip Brna <xbrnaf00>
    ChatGPT

inpired by: - https://farzad-1996.github.io/files/ICCIT_2020_(Finger_Vein).pdf
            - https://ieeexplore.ieee.org/abstract/document/8866626
"""

import numpy as np
import cv2
from scipy import ndimage

class Preprocessor:
    def __init__(self):
        pass

    def __roi(self, image: np.ndarray) -> np.ndarray:
        """
        Define the region of interest (ROI) in the image.
        
        image: np.ndarray
            The input image on which the region of interest is to be defined.
            It is assumed that the image is a 2D array (grayscale image).
        
        Returns:
            np.ndarray: A binary mask of the region of interest (True for the region, False for the background).
        """
        height, width = image.shape
        roi = np.zeros_like(image, dtype=bool)  # Initialize a mask with the same size as the image
        # Define a square region in the center of the image
        roi[height // 4:3 * height // 4, width // 4:3 * width // 4] = True
        return roi

    def detect_finger_vein_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Detect the finger vein mask by applying thresholding and morphological operations.
        
        image: np.ndarray
            The input grayscale image of the finger (2D array).
        
        Returns:
            np.ndarray: A binary mask where the finger vein region is white (255) and the background is black (0).
        """
        # Apply the region of interest (ROI) mask to the image
        thresh = self.__roi(image)
        binary = np.zeros_like(image, dtype=bool)
        # Create a binary mask where the image pixel values are greater than the threshold
        binary[image > thresh] = True

        # Convert the binary mask to uint8 format (255 for True, 0 for False)
        binary = binary.astype(np.uint8) * 255
        # Perform morphological opening to remove small unwanted regions
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

        # Fill holes in the binary mask to get a complete vein mask
        filled_mask = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

        return filled_mask

    def preprocess_image(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the input image by generating a vein mask, applying the mask, 
        and enhancing the image through various steps including CLAHE, 
        noise reduction, and sharpening.
        
        image: np.ndarray
            The input image (grayscale), which will undergo preprocessing to enhance vein patterns.
        
        Returns:
            tuple: A tuple containing the following preprocessed images:
                - vein_mask: np.ndarray - The binary mask of the detected vein region.
                - masked_image: np.ndarray - The input image with the vein mask applied to isolate the vein area.
                - clahe_image: np.ndarray - The image after applying Contrast Limited Adaptive Histogram Equalization (CLAHE).
                - blurred_image: np.ndarray - The image after applying Gaussian Blur for noise reduction.
                - sharp_image: np.ndarray - The image after applying unsharp masking to enhance details.
        """
        # Generate the vein mask by detecting finger veins
        vein_mask = self.detect_finger_vein_mask(image)
        # Apply the vein mask to the original image to isolate the vein region
        masked_image = cv2.bitwise_and(image, image, mask=vein_mask)
        
        # Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_image = clahe.apply(masked_image)

        # Apply median filter to reduce noise
        median_filtered_image = cv2.medianBlur(clahe_image, 5)
        
        # Apply Gaussian Blur to further reduce noise
        blurred_image = cv2.GaussianBlur(median_filtered_image, (5, 5), 0)
        
        # Apply unsharp masking to enhance the image's details
        sharp_image = cv2.addWeighted(clahe_image, 1.3, blurred_image, -0.3, 0)
                
        return vein_mask, masked_image, clahe_image, blurred_image, sharp_image
