"""
BIO Project 2024
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
    ClaudeAI
"""

import numpy as np
import cv2
from src.Postprocessor import Postprocessor

class FeatureExtractor:
    def __init__(self, vein_image: np.ndarray):
        """
        Initializes the FeatureExtractor with a preprocessed vein image and prepares it for feature extraction.

        vein_image: np.ndarray
            A binary or grayscale image representing the vein structure. The image should highlight vein 
            lines clearly, with veins typically represented in white on a dark background.
        """
        # Store the provided vein image
        self.vein_image = vein_image
        self.WHITE = 255  # Constant for white pixel value
        self.BLACK = 0    # Constant for black pixel value

        # Initialize the Postprocessor for skeletonization and artifact removal
        self.postprocessor = Postprocessor()
        
        # Apply skeletonization on the vein image to reduce it to a single-pixel-wide skeleton
        self.skeleton = self.postprocessor.skeletonize(self.vein_image)

        # Remove artifacts from the skeleton with a threshold of 30 pixels
        self.removed_artefacts = self.postprocessor.remove_artefacts(self.skeleton, 30)

    def get_skeleton(self) -> np.ndarray:
        """
        Returns the skeletonized version of the vein image.

        Returns:
            np.ndarray: The skeletonized version of the input vein image, where veins are reduced to 
                        single-pixel width lines.
        """
        return self.skeleton
    
    def get_image(self) -> np.ndarray:
        """
        Returns the original vein image as provided during initialization.

        Returns:
            np.ndarray: The original vein image.
        """
        return self.vein_image

    def extract_bifurcations(self) -> list:
        """
        Detects and identifies bifurcations (branching points) and crossings in the skeleton image.

        Returns:
            list: A list of bifurcation points. Each point is represented as a list [y, x, shape, direction],
                  where 'y' and 'x' are the row and column coordinates, 'shape' is a string indicating the type 
                  of bifurcation or crossing, and 'direction' is the orientation in degrees.
        """

        def is_bifurcation_or_crossing_shape(neighborhood: np.ndarray) -> tuple:
            """
            Determines if a 3x3 pixel neighborhood matches a bifurcation or crossing pattern.

            neighborhood: np.ndarray
                A 3x3 region of pixels centered around the current pixel in the skeleton image.
            
            Returns:
                tuple: A tuple (match, shape, direction) where:
                       - match (bool): True if the neighborhood matches a bifurcation or crossing pattern.
                       - shape (str): Type of bifurcation ("bY" for Y-shape, "bT" for T-shape) or "c" for crossing.
                       - direction (int): Orientation angle of the bifurcation or crossing in degrees 
                                          (0, 90, 180, 270).
            """
            # Define binary patterns for Y-shaped and T-shaped bifurcations
            patterns = [
                np.array([[False, False, False], [False, True, False], [True, False, True]]),  # Y shape (0°)
                np.array([[False, False, True], [False, True, False], [False, False, True]]),  # Y shape (90°)
                np.array([[True, False, True], [False, True, False], [False, False, False]]),  # Y shape (180°)
                np.array([[True, False, False], [False, True, False], [True, False, False]]),  # Y shape (270°)
                np.array([[True, True, True], [False, True, False], [False, False, False]]),    # T shape (0°)
                np.array([[False, False, True], [False, True, True], [False, False, True]]),    # T shape (90°)
                np.array([[False, False, False], [False, True, False], [True, True, True]]),    # T shape (180°)
                np.array([[True, False, False], [True, True, False], [True, False, False]]),    # T shape (270°)
                np.array([[True, False, True], [False, True, False], [True, False, True]]),     # X shape (crossing)
                np.array([[False, True, False], [True, True, True], [False, True, False]])      # + shape (crossing)
            ]

            neighborhood_mask = (neighborhood == self.WHITE)  # Mask for white pixels in the neighborhood

            for i, pattern in enumerate(patterns):
                # Check if the current neighborhood matches one of the patterns
                if np.all(neighborhood_mask[pattern]):
                    direction = (i % 4) * 90  # Calculate rotation angle in degrees
                    if i <= 3:
                        return True, 'bY', direction  # Y-shaped bifurcation
                    elif 4 <= i <= 7:
                        return True, 'bT', direction  # T-shaped bifurcation
                    else:
                        return True, 'c', direction   # Crossing point
                
            return False, None, None  # No match found

        # Initialize a list to store bifurcation points
        bifurcation_points = []

        # Iterate over each pixel in the skeleton image
        for y in range(1, self.skeleton.shape[0] - 1):
            for x in range(1, self.skeleton.shape[1] - 1):
                # Extract the 3x3 neighborhood around the pixel
                neighborhood = self.skeleton[y-1:y+2, x-1:x+2]

                # Count the number of white pixels in the neighborhood
                num_white_pixels = np.sum(neighborhood == self.WHITE)

                # Check if the pixel could be a bifurcation or crossing
                if 3 <= num_white_pixels <= 6:
                    match, shape, direction = is_bifurcation_or_crossing_shape(neighborhood)
                    if match:
                        bifurcation_points.append([y, x, shape, direction])
        
        return bifurcation_points

    def extract_endpoints(self) -> list:
        """
        Identifies endpoints in the skeletonized vein image, where vein lines terminate. 
        Endpoints are pixels in the skeleton with exactly one connected neighbor.
        
        Returns:
            list: A list of coordinates for the endpoints, with each endpoint represented 
                  as a list [y, x] where 'y' is the row and 'x' is the column of the endpoint pixel.
        """
        # Use the skeleton with removed artifacts for endpoint detection
        skeleton = self.removed_artefacts

        # List to store the coordinates of detected endpoints
        endpoints = []

        # Iterate over each pixel in the skeletonized image (avoid boundary pixels)
        y = 1
        while y < skeleton.shape[0] - 1:
            x = 1
            while x < skeleton.shape[1] - 1:
                # Extract the 3x3 neighborhood centered on the current pixel
                neighborhood = skeleton[y-1:y+2, x-1:x+2].flatten()

                # Count the number of white (vein) pixels in the neighborhood
                neighbor_count = np.sum(neighborhood == self.WHITE)

                # If exactly one neighboring pixel is part of the skeleton, this pixel is an endpoint
                if neighbor_count == 1:
                    endpoints.append([y, x])  # Record the coordinates of the endpoint
                    x += 2  # Move to the next region to avoid redundant checks
                x += 1
            y += 2
        return endpoints

    def extract_local_histograms(self, patch_size: tuple = (50, 50)) -> list:
        """
        Divides the vein image into non-overlapping patches and calculates the intensity histogram for each patch.

        patch_size: tuple
            A tuple representing the size (height, width) of each patch in pixels.
        
        Returns:
            list: A list of histograms, with each histogram representing the intensity 
                  distribution within a patch. Each histogram corresponds to one patch in the image.
        """
        # Dimensions of the original vein image and patch size
        height, width = self.vein_image.shape
        patch_height, patch_width = patch_size

        # List to store histograms for each patch
        histograms = []

        # Loop over the image, dividing it into patches
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                # Extract a patch from the vein image
                patch = self.vein_image[y:y + patch_height, x:x + patch_width]

                # Ignore patches that are smaller than the defined size (at image boundaries)
                if patch.shape[0] != patch_height or patch.shape[1] != patch_width:
                    continue

                # Calculate the intensity histogram for the current patch
                hist = cv2.calcHist([patch], [0], None, [256], [0, 256])
                histograms.append(hist)  # Add histogram to the list
        return histograms

    def get_features(self) -> dict:
        """
        Creates a comprehensive feature descriptor for the vein image by combining multiple features:
        bifurcations, endpoints, local histograms, and structural information.

        Returns:
            dict: A dictionary containing various vein feature descriptors:
                - 'bifurcations': List of bifurcation points and their properties.
                - 'endpoints': List of endpoints coordinates.
                - 'localHistograms': List of intensity histograms for each image patch.
                - 'maxCurvature': The original vein image with maximum curvature details.
                - 'skeleton': The skeletonized vein image.
        """
        # Extract individual features from the image
        bifurcations = self.extract_bifurcations()  # Includes crossings as part of bifurcations
        endpoints = self.extract_endpoints()  # Find endpoints in the vein structure
        histograms = self.extract_local_histograms()  # Compute histograms for patches

        # Combine all features into a comprehensive descriptor dictionary
        descriptor = {
            'bifurcations': bifurcations,
            'endpoints': endpoints,
            'localHistograms': histograms,
            'maxCurvature': self.vein_image,  # Original vein image for reference
            'skeleton': self.skeleton  # Skeletonized image
        }
        
        return descriptor
