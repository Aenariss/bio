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

def pipeline(image, intermediate: bool=False):
    """
    Perform the entire image processing pipeline, including loading the image, enhancing its quality,
    removing noise, and returning the processed features. Optionally return intermediate images for debugging or analysis.
    
    image: np.ndarray or str
        The input image. Can be a NumPy array (already loaded image) or a string (file path to the image).
        
    intermediate: bool, optional (default=False)
        If True, returns intermediate images at each step for debugging or analysis. Otherwise, only the final result is returned.
    
    Returns:
        tuple:
            - The processed image, with artefacts removed and the veins isolated.
            - The vein mask, without the first few columns (if intermediate=False).
            
        If `intermediate=True`, also returns:
            - Original image
            - The vein mask
            - The masked image after vein isolation
            - The CLAHE-enhanced image
            - The blurred image (after noise reduction)
            - The sharp image (after enhancement)
            - The result of max curvature computation
            - The normalized and thresholded result images
            - The final image after removing edge artefacts
    """
    # Check if image is a file path (string) and load accordingly
    if isinstance(image, str):
        image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # If image is a byte stream (np.ndarray), decode it
    elif isinstance(image, np.ndarray):
        image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)

    # Raise error if image is not loaded correctly
    if image is None:
        raise ValueError("Image not loaded. Check the file path or integrity.")
    
    # Instantiate preprocessor and postprocessor objects
    preprocessor = Preprocessor()
    postprocessor = Postprocessor()

    # Step 1: Preprocess the image
    # Apply preprocessing pipeline (e.g., vein detection, enhancement)
    vein_mask, masked_image, clahe_image, blurred_image, sharp_image = preprocessor.preprocess_image(image)

    # Step 2: Apply max curvature processing
    result = MaxCurvature().max_curvature(sharp_image, vein_mask, sigma=8)

    # Step 3: Binarize the result to extract vein features
    result_normalized, result_thresholded = postprocessor.binarize(result, intermediate=True)

    # Step 4: Apply morphological operations to connect broken veins
    result_connected_veins = postprocessor.additional_vein_connections(result_thresholded)

    # Step 5: Remove edge artefacts around the mask area
    result_no_mask_edge = postprocessor.remove_mask_edge(result_connected_veins, vein_mask)

    # If intermediate is True, return the intermediate steps for analysis
    if intermediate:
        return image, vein_mask, masked_image, clahe_image, blurred_image, sharp_image, result, result_normalized, result_no_mask_edge

    # Return the final processed image and the vein mask with edges removed
    return result_no_mask_edge, vein_mask[:, 4:]

