"""
BIO Project 2024
Author: 
    Filip Brna <xbrnaf00>
    Vojtech Fiala <xfiala61>
    ChatGPT
File showing an example pipeline to obtain features from a grayscale finger photo
"""

import cv2
import numpy as np
from src.Preprocessor import Preprocessor
from src.Postprocessor import Postprocessor
from src.MaxCurvature import MaxCurvature

def pipeline(image, intermediate=False):
    """
    Do all the necessary steps to process an image - load it, enhance its quality, remove noise and return the features. Optionally get all intermediate images
    """
    # Path to file
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Image is a np array
    elif isinstance(image, np.ndarray):
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # Error
    if image is None:
        raise ValueError("Image not loaded. Check the file path or integrity.")
    
    preprocessor = Preprocessor()
    postprocessor = Postprocessor()


    # Preprocess the image
    vein_mask, masked_image, clahe_image, blurred_image, sharp_image = preprocessor.preprocess_image(image)
    # do the max curvature
    result = MaxCurvature().max_curvature(sharp_image, vein_mask, sigma=8)

    result_normalized, result_thresholded = postprocessor.binarize(result, intermediate=True)

    '''
    #TODO:Doesn't look bad, but it's not perfect, slightly worse than fixed threshold
    result_normalized_without_noise = cv2.adaptiveThreshold(result_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 1)
    # invert black and white
    result_normalized_without_noise = cv2.bitwise_not(result_normalized_without_noise, result_normalized_without_noise)
    '''

    result_connected_veins = postprocessor.additional_vein_connections(result_thresholded)
    result_no_mask_edge = postprocessor.remove_mask_edge(result_connected_veins, vein_mask)

    if intermediate:
        return image, vein_mask, masked_image, clahe_image, blurred_image, sharp_image, result, result_normalized, result_no_mask_edge

    return result_no_mask_edge, vein_mask[:, 4:]


