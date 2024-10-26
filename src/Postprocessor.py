"""
BIO Project 2024

This module contains a class to apply postprocessing to an image 
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
"""

import numpy as np
import cv2
from skimage import measure, morphology


class Postprocessor:
    def __init__(self):
        pass

    def skeletonize(self, image):

        skeleton = morphology.skeletonize(image).astype(np.uint8)

        return self.binarize(skeleton)
    
    def remove_artefacts(self, image, removal_length=30):
        """
        removal_length specifies the minimal length of vein to be consdiered as such, everything shorter is an artefact
        """

        # Label connected components in the skeletonized image
        labels = measure.label(image, connectivity=2)

        # Create an output image to store the filtered components
        filtered_image = np.zeros_like(image)

        # Iterate over each connected component and measure length
        for region in measure.regionprops(labels):
            # Calculate the actual length of each line segment
            line_length = len(region.coords)
            
            # Keep only components with a length greater than removal_length
            if line_length >= removal_length:
                for coord in region.coords:
                    filtered_image[coord[0], coord[1]] = 255

        return filtered_image
    
    def binarize(self, image, threshold=3, intermediate=False):

        # First normalize the image
        result_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))

        # Scale the normalized result
        result_scaled = (result_normalized * 255).astype(np.uint8)
        
        # Fixed threshold seems to give best results
        _, result_thresholded = cv2.threshold(result_scaled, threshold, 255, cv2.THRESH_BINARY)

        if intermediate:
            return result_normalized, result_thresholded
        
        return result_thresholded
    
    def additional_vein_connections(self, image):

        # Morphological operations
        kernel = np.ones((3,3), np.uint8)
        
        # Apply closing to connect broken veins
        result_normalized_without_noise = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

        return result_normalized_without_noise
    
    def remove_mask_edge(self, image, vein_mask):


        kernel = np.ones((3,3), np.uint8)

        # make mask a little bit smaller to remove the edges
        vein_mask = cv2.erode(vein_mask, kernel, iterations=3)
        removed_edge_veins = cv2.bitwise_and(image, vein_mask)

        # also remove first few columns of image to fix artefacts
        removed_edge_veins = removed_edge_veins[:, 4:] 

        return removed_edge_veins
        

