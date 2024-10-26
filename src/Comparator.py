"""
BIO Project 2024

This module contains a class to compare results 
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
    Gemini
"""

import numpy as np
import cv2
from src.DataLoader import DataLoader
from src.Pipeline import pipeline
from src.FeatureExtractor import FeatureExtractor

from scipy.spatial.distance import cdist
from skimage.feature import hog
from scipy.spatial.distance import cosine

class Comparator:
    def __init__(self, threshold=14):
        self.threshold = threshold

    def __phase_correlation(self, img1, img2):
        """
        Phase correlation to compute the shift between two images and measure similarity.
        Uses Fourier transforms for translation-invariant matching.
        """
        # Convert images to float32 for Fourier transform
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Perform Fourier transform on both images
        dft1 = np.fft.fft2(img1)
        dft2 = np.fft.fft2(img2)

        # Compute cross-power spectrum
        conj_dft2 = np.conj(dft2)
        cross_power_spectrum = (dft1 * conj_dft2) / (np.abs(dft1 * conj_dft2) + 1e-8)

        # Inverse Fourier transform to get phase correlation
        inverse_dft = np.fft.ifft2(cross_power_spectrum)
        correlation_result = np.abs(inverse_dft)

        # Peak value in correlation result indicates the degree of alignment
        max_corr_value = np.max(correlation_result)

        # Scale the score to make it readable and inverse it to make it lower the better
        result_score = 100 - int(max_corr_value * 1000)

        result_score = 0 if result_score < 0 else result_score
        
        return result_score

    '''
    def multi_scale_comparison(self, img1, img2):
        """
        A multi-scale comparison approach to handle variations in scale by downsampling
        and calculating similarity at different resolutions.
        """
        scales = [1.0, 0.9, 0.8]  # Rescale factors to handle slight scale differences
        scores = []
        
        for scale in scales:
            # Resize images to the same size for each scale
            img1_resized = cv2.resize(img1, (0, 0), fx=scale, fy=scale)
            img2_resized = cv2.resize(img2, (0, 0), fx=scale, fy=scale)

            # Get the phase correlation score at this scale
            score = self.phase_correlation(img1_resized, img2_resized)
            scores.append(score)

        return max(scores)  # Return the best score across scales

    def compare(self, img1, img2):
        """
        Compare two vein images using the revised strategy.
        Preprocessing, multi-scale matching, and phase correlation.
        """
        # Perform multi-scale comparison for robustness to scale changes
        final_score = self.multi_scale_comparison(img1, img2)

        # Determine if the match passes the threshold
        return (final_score >= self.threshold, final_score)
    '''

    '''
    def compare(self, img1, img2):
        """
        Compute phase correlation between two images.
        Returns the matching score (a higher score means better alignment/matching).
        """
        # Convert images to float32
        img1 = img1.astype(np.float32)
        img2 = img2.astype(np.float32)

        # Perform the Fourier transform on both images
        dft1 = np.fft.fft2(img1)
        dft2 = np.fft.fft2(img2)

        # Compute the cross-power spectrum
        conj_dft2 = np.conj(dft2)
        cross_power_spectrum = (dft1 * conj_dft2) / (np.abs(dft1 * conj_dft2) + 1e-8)

        # Perform inverse Fourier transform on cross-power spectrum to get phase correlation
        inverse_dft = np.fft.ifft2(cross_power_spectrum)
        correlation_result = np.abs(inverse_dft)

        # Get the peak value in the correlation result
        max_corr_value = np.max(correlation_result)

        result_score = max_corr_value * 1000 # * 1000 to make it more readable

        return (result_score >= self.threshold, result_score)
    '''
    
    
    def compare_all(self, compared_img_path):
        """
        Compare multiple pictures in a set to a given picture and get the match results
        """
        results = []
        data = DataLoader().load_images()
        original_features = pipeline(compared_img_path)
        original_descriptor = FeatureExtractor(original_features).create_descriptor()

        for person in data:
            for finger in data[person]:
                for photo in data[person][finger]:
                    current_features = pipeline(photo)
                    cmp_descriptor = FeatureExtractor(current_features).create_descriptor()
                    result_score = self.compare_descriptors(original_features, current_features, original_descriptor, cmp_descriptor)
                    print(result_score)
                    results.append(result_score)
        print(results)

    # Comparison of two arrays of coordinates (such as bifurcations) 
    def __compare_bifurcations(self, img1, img2, b1, b2):

        b1 = np.array(b1)
        b2 = np.array(b2)

        def compute_hog_descriptors(image, bifurcations, patch_size=16):
            hog_descriptors = []
            for (y, x) in bifurcations:
                patch = image[y-patch_size//2:y+patch_size//2, x-patch_size//2:x+patch_size//2]
                if patch.shape == (patch_size, patch_size):
                    hog_desc = hog(patch, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
                    hog_descriptors.append(hog_desc)
            return np.array(hog_descriptors)

        # Example usage
        descriptors_img1 = compute_hog_descriptors(img1, b1)
        descriptors_img2 = compute_hog_descriptors(img2, b2)

        # Compare HOG descriptors with cosine similarity
        similarity_matrix = cdist(descriptors_img1, descriptors_img2, metric='cosine')
        min_distances = np.min(similarity_matrix, axis=1)
        similarity_score = np.mean(min_distances)

        return int(similarity_score * 1000) # the lower the better, UNUSABLE on its own but gives results that might be used together with some other features


    # Method to compare two descriptors
    def compare_descriptors(self, img1, img2, descriptor1, descriptor2):
        """
        Compares two descriptors and returns a similarity score.
        """
        score = 0

        # compare bifurcations using hog descriptors
        score += self.__compare_bifurcations(img1, img2, descriptor1['bifurcations'], descriptor2['bifurcations'])

        # phase correlation between images
        score += self.__phase_correlation(img1, img2)
        
        return score  # The lower the score, the more similar the two descriptors

