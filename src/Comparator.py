"""
BIO Project 2024

This module contains a class to compare results 
Author: Vojtech Fiala <xfiala61> + ChatGPT + Gemini
"""

import numpy as np
from src.DataLoader import DataLoader
from src.Pipeline import pipeline

class Comparator:
    def __init__(self, threshold=14):
        self.threshold = threshold


    '''
    New compare method which has better values for correct fingers, but also some false positives for treshold 18
    def phase_correlation(self, img1, img2):
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

        # Scale the score to make it readable (as in your original)
        result_score = max_corr_value * 1000
        
        return result_score

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

    def compare(self, img1, img2):
        """
        Compute phase correlation between two images.
        Returns the matching score (a higher score means better alignment/matching).
        Much better than regular Miura Match
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
    
    
    def compare_all(self, compared_img_path):
        """
        Compare multiple pictures in a set to a given picture and get the match results
        """
        results = []
        data = DataLoader().load_images()
        original_features = pipeline(compared_img_path)

        for person in data:
            for finger in data[person]:
                for photo in data[person][finger]:
                    current_features = pipeline(photo)
                    result_score = self.compare(current_features, original_features)
                    results.append(result_score)
        print(results)
