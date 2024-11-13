"""
BIO Project 2024
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
"""

import numpy as np
import cv2
from src.Postprocessor import Postprocessor

class FeatureExtractor:
    def __init__(self, vein_image):
        # Initialize with the preprocessed vein image (binary or grayscale)
        self.vein_image = vein_image
        self.WHITE = 255
        self.BLACK = 0
        self.postprocessor = Postprocessor()
        self.skeleton = self.postprocessor.skeletonize(self.vein_image)

        # Will this be needed? If so, should we do it on skeleton or the original vein image?
        self.removed_artefacts = self.postprocessor.remove_artefacts(self.skeleton, 30)

    def get_skeleton(self):
        return self.skeleton
    
    def get_image(self):
        return self.vein_image

    # Method to extract bifurcations and crossings
    # Because of how our source looks, this may be less accurate than we'd like it to be... still better than nothing ig
    def extract_bifurcations(self):
        """
        This method detects and returns the bifurcations (branching points)
        and crossings of the veins in the image.
        """

        WHITE = self.WHITE

        def is_bifurcation_or_crossing_shape(neighborhood):

            # Check for T or Y shapes by checking specific patterns
            # Explicitly state possible pattersn
            # True states White pixel, False states Black (i dont care about) pixel
            patterns = [
                np.array([[False, False, False], [False, True, False], [True, False, True]]), # Y
                np.array([[False, False, True], [False, True, False], [False, False, True]]), # Y 90 degress
                np.array([[True, False, True], [False, True, False], [False, False, False]]), # Y 180 degrees
                np.array([[True, False, False], [False, True, False], [True, False, False]]), # Y 270 degress
                np.array([[True, True, True], [False, True, False], [False, False, False]]), # T
                np.array([[False, False, True], [False, True, True], [False, False, True]]), # T 90 degrees
                np.array([[False, False, False], [False, True, False], [True, True, True]]), # T 180 degrees
                np.array([[True, False, False], [True, True, False], [True, False, False]]), # T 270 degrees
                np.array([[True, False, True], [False, True, False], [True, False, True]]),  # Crossing X
                np.array([[False, True, False], [True, True, True], [False, True, False]]) # Crossing +
            ]

            pattern_lengths = len(patterns)
            neighborhood_mask = (neighborhood == self.WHITE)

            # Check bifurcation patterns
            for i in range(pattern_lengths):

                # Pattern matches
                if np.all(neighborhood_mask[patterns[i]]):
                    direction = (i % 4) * 90
                    if i <= 3:
                        return True, 'bY', direction
                    elif 4 <= i <= 7:
                        return True, 'bT', direction
                    else:
                        return True, 'c', direction
                
            return False, None, None

            # Compare with each bifurcation pattern, only check white pixels
            #neighborhood_mask = (neighborhood == self.WHITE)
            # If at least one pattern matches all white pixels in the neighborhood, return match
            #return np.any([np.all(neighborhood_mask[pattern]) for pattern in patterns])

        # Create an empty image to mark bifurcations and crossings
        bifurcation_points = []

        # Iterate over each pixel in the thinned image
        y = 1  # Start from the first row to avoid out-of-bounds
        while y < self.skeleton.shape[0] - 1:
            x = 1  # Start from the first column to avoid out-of-bounds
            while x < self.skeleton.shape[1] - 1:
                # Check if the pixel is part of a vein
                # Extract the 3x3 neighborhood
                neighborhood = self.skeleton[y-1:y+2, x-1:x+2] # Choose which is better - skeleton or skeleton w/o artefacts

                # Count the number of white pixels (255) in the neighborhood
                num_white_pixels = np.sum(neighborhood == WHITE)

                # Check if the number of white pixels suggests a potential crossing or bifurcation
                if 3 <= num_white_pixels <= 6:
                    # Check for bifurcation pattern
                    match, shape, direction = is_bifurcation_or_crossing_shape(neighborhood)
                    if match:
                        bifurcation_points.append([y, x, shape, direction])
                        # Move to the next neighborhood by skipping over the relevant area
                        x += 1  # Skip to the next potential neighborhood
                        # y += 1 # check if this woul work better

                x += 1  # Move to the next pixel in the same row
            y += 1  # Move to the next row, this can result in one bifurcation being present several times but should still somehow work
        
        return bifurcation_points  # Return the coordinates of the bifurcation points

    # Method to extract vein endpoints (useful for topological structure)
    def extract_endpoints(self):
        """
        This method identifies the endpoints of veins, where the vein lines terminate.
        """
        # Skeletonize the image (if not already skeletonized)
        skeleton = self.removed_artefacts

        # Get the coordinates of the endpoints (pixels with exactly one neighbor)
        endpoints = []
        y = 1  # Start from the first row to avoid out-of-bounds
        while y < skeleton.shape[0] - 1:
            x = 1  # Start from the first column to avoid out-of-bounds
            while x < skeleton.shape[1] - 1:
                # Check the 8-connectivity (neighbors of the pixel)
                neighborhood = skeleton[y-1:y+2, x-1:x+2].flatten()
                # Count the number of neighbors that are part of the skeleton (value 1)
                neighbor_count = np.sum(neighborhood == self.WHITE)  # Exclude the center pixel
                if neighbor_count == 1:
                    endpoints.append([x, y])
                    x += 2 # move onto the next neighborhood
                x += 1
            y += 2
        return endpoints

    # Method to combine all features
    def get_features(self):
        """
        This method creates a feature descriptor by combining various vein features.
        """
        bifurcations = self.extract_bifurcations() # crossings are part of bifurtcations
        endpoints = self.extract_endpoints()

        # Combine all features into a descriptor
        descriptor = {
            'bifurcations': bifurcations,
            'endpoints': endpoints
        }
        
        return descriptor
