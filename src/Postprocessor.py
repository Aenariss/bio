"""
BIO Project 2024

This module contains a class to apply postprocessing to an image 
Author: 
    Vojtech Fiala <xfiala61>
    Filip Brna <xbrnaf00>
    ChatGPT
"""

import numpy as np
import cv2
from skimage import measure, morphology

class Postprocessor:
    def __init__(self):
        pass

    def skeletonize(self, image: np.ndarray) -> np.ndarray:
        """
        Skeletonizes the input image to reduce the vein structures to a thin line representation.
        
        image: np.ndarray
            A binary image where veins are represented as white (255) and the background as black (0).
            This is an OpenCV image represented as a NumPy array.
        
        Returns:
            np.ndarray: A binary image where the skeletonized veins are represented as white (255) and the background as black (0).
        """
        # Apply skeletonization using skimage's morphology function
        skeleton = morphology.skeletonize(image).astype(np.uint8)
        
        # Binarize the result
        return self.binarize(skeleton)
    
    def remove_artefacts(self, image: np.ndarray, removal_length: int = 30) -> np.ndarray:
        """
        Removes short artefacts (small structures) from the skeletonized image by filtering components based on length.
        
        image: np.ndarray
            A binary image where veins are represented as white (255) and background as black (0).
            This is an OpenCV image represented as a NumPy array.
        
        removal_length: int, optional (default=30)
            Specifies the minimum length of a vein component to be considered as a valid vein. Any components shorter than this length will be removed as artefacts.
        
        Returns:
            np.ndarray: A binary image with artefacts removed, where veins are white (255) and the background is black (0).
        """
        # Label connected components in the binary image
        labels = measure.label(image, connectivity=2)

        # Create an output image to store the filtered components
        filtered_image = np.zeros_like(image)

        # Iterate over each connected component
        for region in measure.regionprops(labels):
            # Calculate the length of each component
            line_length = len(region.coords)
            
            # Keep only components that are longer than the removal length
            if line_length >= removal_length:
                for coord in region.coords:
                    filtered_image[coord[0], coord[1]] = 255

        return filtered_image
    
    def binarize(self, image: np.ndarray, threshold: int = 3, intermediate: bool = False) -> np.ndarray:
        """
        Binarizes the image using a fixed threshold after normalizing it to the range [0, 255].
        
        image: np.ndarray
            The input image to be binarized. This image can be of any intensity range and will be normalized before thresholding.
        
        threshold: int, optional (default=3)
            The threshold value to binarize the image. Pixels greater than the threshold will be set to 255, others to 0.
        
        intermediate: bool, optional (default=False)
            If set to True, returns the normalized image as well as the thresholded image. Otherwise, only the thresholded image is returned.
        
        Returns:
            np.ndarray: The binarized image where pixels above the threshold are white (255) and those below are black (0).
            If `intermediate` is True, also returns the normalized image.
        """
        # Normalize the image to the range [0, 1]
        result_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Scale the normalized result to the range [0, 255]
        result_scaled = (result_normalized * 255).astype(np.uint8)
        
        # Apply thresholding to the scaled image
        _, result_thresholded = cv2.threshold(result_scaled, threshold, 255, cv2.THRESH_BINARY)

        # If intermediate is True, return both normalized and thresholded images
        if intermediate:
            return result_normalized, result_thresholded
        
        # Return only the thresholded image
        return result_thresholded
    
    def additional_vein_connections(self, image: np.ndarray) -> np.ndarray:
        """
        Connect broken vein structures by applying morphological closing to the image.
        
        image: np.ndarray
            The input binary image where veins are represented as white (255) and the background as black (0).
        
        Returns:
            np.ndarray: The processed image with additional vein connections, where veins are white (255) and the background is black (0).
        """
        # Define a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Apply morphological closing to connect small gaps in veins
        result_normalized_without_noise = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return result_normalized_without_noise
    
    def remove_mask_edge(self, image: np.ndarray, vein_mask: np.ndarray) -> np.ndarray:
        """
        Removes the edges of the mask, primarily to avoid artefacts around the border of the image.
        
        image: np.ndarray
            The input image where veins are represented as white (255) and the background as black (0).
        
        vein_mask: np.ndarray
            A binary mask where veins are represented as white (255) and the background as black (0).
            This mask is applied to the image to focus on the vein region.
        
        Returns:
            np.ndarray: The image with the edges of the mask removed, focusing on the vein region.
        """
        # Define a kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)

        # Erode the mask to shrink it slightly and remove the edges
        vein_mask = cv2.erode(vein_mask, kernel, iterations=3)
        
        # Apply the modified mask to the image to remove the edges
        removed_edge_veins = cv2.bitwise_and(image, vein_mask)

        # Also remove the first few columns of the image to fix artefacts near the image border
        removed_edge_veins = removed_edge_veins[:, 4:]

        return removed_edge_veins
