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
        self.postprocessor = Postprocessor()
        self.skeleton = self.postprocessor.skeletonize(self.vein_image)

        # Will this be needed? If so, should we do it on skeleton or the original vein image?
        self.removed_artefacts = self.postprocessor.remove_artefacts(self.skeleton, 30)

    # Method to extract bifurcations and crossings
    # Because of how our source looks, this may be less accurate than we'd like it to be... still better than nothing ig
    def extract_bifurcations(self):
        """
        This method detects and returns the bifurcations (branching points)
        and crossings of the veins in the image.
        """

        # Create an empty image to mark bifurcations and crossings
        bifurcations = np.zeros_like(self.skeleton, dtype=np.uint8)
        crossings = np.zeros_like(self.skeleton, dtype=np.uint8)

        # Iterate over each pixel in the thinned image
        for y in range(1, self.skeleton.shape[0] - 1):
            for x in range(1, self.skeleton.shape[1] - 1):
                # Check if the pixel is part of a vein
                if self.skeleton[y, x] == self.WHITE:
                    # Extract the neighborhood
                    neighborhood = self.skeleton[y-1:y+2, x-1:x+2]
                    # Count the number of white pixels (255) in the neighborhood
                    num_white_pixels = np.sum(neighborhood == self.WHITE)
                    
                    # Check for bifurcation (3 connections) and crossing (4 connections)
                    if num_white_pixels >= 4:
                        # Mark as crossing
                        crossings[y, x] = self.WHITE
                    elif num_white_pixels == 3:
                        # Mark as bifurcation
                        bifurcations[y, x] = self.WHITE
                        
        # Get coordinates of bifurcations
        bifurcation_points = np.column_stack(np.where(bifurcations == 255))

        # Get coordinates of crossings
        crossing_points = np.column_stack(np.where(crossings == 255))
        
        return bifurcation_points, crossing_points  # Return the coordinates of the bifurcation points
    
    # Method to extract vein curvature
    def extract_curvature(self):
        """
        This method computes the local curvature at each point of the veins.
        """
        # Placeholder: implement curvature extraction
        curvature_map = np.zeros_like(self.vein_image)  # Dummy value
        return curvature_map

    # Method to extract vein thickness
    def extract_thickness(self):
        """
        This method calculates the thickness of each vein in the image.
        """
        # Placeholder: implement vein thickness extraction
        thickness_map = np.zeros_like(self.vein_image)  # Dummy value
        return thickness_map

    # Method to extract vein orientation (direction of veins)
    def extract_orientation(self):
        """
        This method calculates the orientation of the veins in the image.
        """
        # Placeholder: implement vein orientation extraction
        orientation_map = np.zeros_like(self.vein_image)  # Dummy value
        return orientation_map

    # Method to extract vein density
    def extract_density(self):
        """
        This method computes the vein density, indicating how closely veins are packed.
        """
        # Placeholder: implement vein density extraction
        density_map = np.zeros_like(self.vein_image)  # Dummy value
        return density_map

    # Method to extract vein endpoints (useful for topological structure)
    def extract_endpoints(self):
        """
        This method identifies the endpoints of veins, where the vein lines terminate.
        """
        # Placeholder: implement vein endpoint detection
        endpoints = np.array([])  # Dummy value
        return endpoints

    # Method to create a full descriptor combining all features
    def create_descriptor(self):
        """
        This method creates a feature descriptor by combining various vein features.
        """
        bifurcations, crossings = self.extract_bifurcations()
        curvature = self.extract_curvature()
        thickness = self.extract_thickness()
        orientation = self.extract_orientation()
        density = self.extract_density()
        endpoints = self.extract_endpoints()

        # Combine all features into a descriptor
        descriptor = {
            'bifurcations': bifurcations,
            'crossings': crossings,
            'curvature': curvature,
            'thickness': thickness,
            'orientation': orientation,
            'density': density,
            'endpoints': endpoints
        }
        
        return descriptor
