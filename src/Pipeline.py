"""
BIO Project 2024
Author: Filip Brna <xbrnaf00>, Vojtech Fiala <xfiala61>, ChatGPT
File showing an example pipeline to obtain features from a grayscale finger photo
"""

import cv2
import numpy as np
from src.Preprocessor import Preprocessor
from src.MaxCurvature import MaxCurvature

def pipeline(image, intermediate=False):
    """
    Do all the necessary steps to process an image - load it, enhance its quality, remove noise and return the features. Optionally get all intermediate images
    """
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError("Image not loaded. Check the file path or integrity.")

    # Preprocess the image
    vein_mask, masked_image, clahe_image, blurred_image, sharp_image = Preprocessor().preprocess_image(image)
    # do the max curvature
    result = MaxCurvature().max_curvature(sharp_image, vein_mask, sigma=8)

    result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))
    result_scaled = (result_normalized * 255).astype(np.uint8)
    
    # Fixed threshold seems to give best results
    _, result_thresholded = cv2.threshold(result_scaled, 3, 255, cv2.THRESH_BINARY)

    '''
    #TODO:Doesn't look bad, but it's not perfect, slightly worse than fixed threshold
    result_normalized_without_noise = cv2.adaptiveThreshold(result_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    # invert black and white
    result_normalized_without_noise = cv2.bitwise_not(result_normalized_without_noise, result_normalized_without_noise)
    '''

    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    
    # Apply closing to connect broken veins
    result_normalized_without_noise = cv2.morphologyEx(result_thresholded, cv2.MORPH_CLOSE, kernel)

    # do mask a little bit smaller to remove the edges
    #vein_mask = cv2.erode(vein_mask, kernel, iterations=3)
    #result_normalized_without_noise = cv2.bitwise_and(result_normalized_without_noise, vein_mask)


    if intermediate:
        return image, vein_mask, masked_image, clahe_image, blurred_image, sharp_image, result, result_normalized, result_normalized_without_noise

    return result_normalized_without_noise

