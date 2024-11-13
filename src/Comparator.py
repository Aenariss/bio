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
from skimage.metrics import structural_similarity
from scipy.spatial.distance import cosine
from scipy.signal import correlate2d

class Comparator:
    def __init__(self, threshold=14):
        self.threshold = threshold

    def __align_images(self, image1, image2, mask1, mask2):
        """Align images using Enhanced Correlation Coefficient (ECC) method"""
        # Define the motion model
        warp_mode = cv2.MOTION_EUCLIDEAN  # Supports translation + rotation
            
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)
            
        # Initialize the warp matrix
        warp_matrix = np.eye(2, 3, dtype=np.float32)
            
        _, warp_matrix = cv2.findTransformECC(
                mask1, 
                mask2,
                warp_matrix, 
                warp_mode,
                criteria,
                inputMask=None,
                gaussFiltSize=1
        )
                
        img2_binary_aligned = cv2.warpAffine(image2, warp_matrix, (image1.shape[1], image1.shape[0]),
                                                flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
                                                borderMode=cv2.BORDER_CONSTANT)

        return img2_binary_aligned


    # Very powerful, could be used on its own with some false negatives
    def __local_histogram_comparison(self, vein_image_1, vein_image_2, patch_size=(50, 50)):
        """
        Compare local histograms of two vein pattern images. The function divides both images into
        smaller patches and compares their histograms at each corresponding location.
        
        Parameters:
            vein_image_1: First vein pattern image (grayscale).
            vein_image_2: Second vein pattern image (grayscale).
            patch_size: Size of the patches to divide the images into (default is 50x50).
        
        Returns:
            Normalized similarity score (0-100) where 0 means perfect match and 100 means no match.
        """

        # Get image dimensions
        height, width = vein_image_1.shape
        patch_height, patch_width = patch_size

        # Initialize a list to hold similarity scores for each patch
        similarity_scores = []

        # Loop over the images and divide them into patches
        for y in range(0, height, patch_height):
            for x in range(0, width, patch_width):
                # Define the patch region in both images
                patch1 = vein_image_1[y:y + patch_height, x:x + patch_width]
                patch2 = vein_image_2[y:y + patch_height, x:x + patch_width]

                # Skip patches that are smaller than the desired patch size (at image boundaries)
                if patch1.shape[0] != patch_height or patch1.shape[1] != patch_width:
                    continue

                # Calculate histograms for both patches
                hist1 = cv2.calcHist([patch1], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([patch2], [0], None, [256], [0, 256])

                # Normalize the histograms
                hist1 = cv2.normalize(hist1, hist1).flatten()
                hist2 = cv2.normalize(hist2, hist2).flatten()

                # Compare the histograms using correlation
                correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

                # Store the similarity score for this patch
                similarity_scores.append(correlation)

        # Calculate the average similarity score across all patches
        avg_similarity = np.mean(similarity_scores)

        # Normalize the similarity score to range [0, 100000], where 0 is a perfect match... values have been empirically observed to be only <100  
        normalized_score = 100000 * (1 - avg_similarity)

        return normalized_score
    
    # https://en.wikipedia.org/wiki/Structural_similarity_index_measure
    def __ssim_comparison(self, img1, img2):
        """
        measures image similarity by comparing three main components: 
            luminance, contrast, and structure between two images. 
        It captures perceived visual quality by focusing on pixel intensity patterns rather than absolute differences
        """
        # Compute SSIM between two images
        ssim_score, _ = structural_similarity(img1, img2, full=True)
        
        # Normalize the SSIM score to a range [0, 100]
        normalized_ssim_score = 100 * (1 - ssim_score)
        return normalized_ssim_score


    def compare_all(self, compared_img_path):
        """
        Compare multiple pictures in a set to a given picture and get the match results
        """
        results = []
        loader = DataLoader(original_finger=compared_img_path)
        data = loader.load_images()
        original_id_person, original_finger, original_id_finger = loader.get_original_image_data()
        original_veins, original_mask = pipeline(compared_img_path)
        original_featuresExtractor = FeatureExtractor(original_veins)
        original_features = original_featuresExtractor.get_features()

        print("Starting comparison of person {} finger {} photo {} with everyone else, the lower the score the better the match...".format(original_id_person, original_finger, original_id_finger))
        for person in data:
            print("Starting comparison of person {} and {}...".format(original_id_person, person))
            for finger in data[person]:
                print("Starting comparison with finger {} belonging to person {}...".format(finger, person))
                for photo in data[person][finger]:
                    current_veins, current_mask = pipeline(photo)
                    current_veins = self.__align_images(original_veins, current_veins, original_mask, current_mask)

                    cmp_features = FeatureExtractor(current_veins)
                    cmp_descriptor = cmp_features.get_features()
                    result_score = self.compare_descriptors(original_veins, current_veins, original_featuresExtractor.get_skeleton(), cmp_features.get_skeleton(), original_features, cmp_descriptor)
                    print("Score: {}".format(result_score))
                    results.append(result_score)
        return results

    # Comparison bifurcations
    def __compare_bifurcations(self, b1, b2):
        
        b1 = sorted(b1, key=lambda item: (item[0], item[1]))
        b2 = sorted(b2, key=lambda item: (item[0], item[1]))

        b1 = np.array(b1)
        b2 = np.array(b2)

        if len(b1) == 0 or len(b2) == 0:
            raise RuntimeError("Could not extract bifurcations! Invalid image!")


        # Extract x, y coordinates from arr1 and arr2
        coords1 = np.array(b1[:, :2], dtype=int)
        coords2 = np.array(b2[:, :2], dtype=int)
        
        # Step 1: Compute all pairwise Euclidean distances between coords1 and coords2
        # (Using broadcasting for fast vectorized distance calculation)
        diff = coords1[:, None, :] - coords2[None, :, :]   # Shape: (len(list1), len(list2), 2)
        distances = np.sqrt(np.sum(diff**2, axis=2))       # Shape: (len(list1), len(list2))
        
        # Step 2: Find the nearest point in list2 for each point in list1 within the threshold
        matches = []
        matched_indices_list2 = set()  # Track matched indices in list2 to avoid duplicate matches

        # Tolerance how distant at most the points can be to even be compared, empirically set to 20
        distance_threshold = 20 

        # maximum penalty possible to serve where normal penalty can;t be calculated (mismatched array sizes...)
        maximum_penalty = distance_threshold + 10 + 5

        for i in range(distances.shape[0]):
            # Find the closest point in list2 for point i in list1
            within_threshold = distances[i] <= distance_threshold
            if np.any(within_threshold):
                # Get the minimum distance and its index among valid matches within the threshold
                min_dist_index = np.argmin(np.where(within_threshold, distances[i], np.inf)) # Take the smallest euclidean value (its index)
                min_distance = distances[i, min_dist_index] # take the two closest
                
                if min_dist_index not in matched_indices_list2:
                    matched_indices_list2.add(min_dist_index)

                    # Extract shapes and directions for comparison
                    shape1, dir1 = b1[i, 2], b1[i, 3]
                    shape2, dir2 = b2[min_dist_index, 2], b2[min_dist_index, 3]
                    
                    # Compare shapes and calculate direction difference
                    shape_match = shape1 == shape2
                    direction_match = dir1 == dir2

                    penalty = min_distance + 10 * int(not shape_match) + 5 * int(not direction_match) # Penalty is the distance + 10 for mismatch of shape + 5 for mismatch of direction
                    matches.append(penalty)
            else:
                # No valid match within the threshold for this point in list1
                matches.append(maximum_penalty)
                

        # Step 3: Handle unmatched points in list2
        additional_penalty = sum([maximum_penalty for j in range(len(coords2)) if j not in matched_indices_list2])

        # Total score, the lower the better
        total_penalty = sum(matches) + additional_penalty

        # Normalize the total_penalty to range [0, 100]
        maximum_possible_penalty = (len(coords1) + len(coords2)) * maximum_penalty
        normalized_penalty = 100 * (total_penalty / maximum_possible_penalty)

        return normalized_penalty

    def __compare_overlap(self, img1, img2):
        # Convert images to float32 before phase correlation
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        # Create binary versions after alignment to preserve original intensity values
        img1_binary = (img1 > 0).astype(np.uint8)
        img2_binary = (img2 > 0).astype(np.uint8)

        overlap = img1_binary * img2_binary
        overlap_count = np.sum(overlap)
        union_count = np.sum(img1_binary) + np.sum(img2_binary) - overlap_count
        overlap_percentage = (overlap_count / union_count) * 100 if union_count != 0 else 0
        
        return 100 - overlap_percentage # how much are they different percentage

    # Method to compare two descriptors
    def compare_descriptors(self, img1, img2, skeleton1, skeleton2, descriptor1, descriptor2):
        """
        Compares two descriptors and returns a similarity score.
        """
        score = 0

        # compare bifurcations using euclidean distance
        score += self.__compare_bifurcations(descriptor1['bifurcations'], descriptor2['bifurcations'])

        # structural similarity, should we do this? kinda works but produces lot of false negatives
        score += self.__ssim_comparison(img1, img2)

        score += self.__local_histogram_comparison(img1, img2)

        score += self.__compare_overlap(skeleton1, skeleton2) # seems usable, some false negatives... maybe remove later

        # normalize the score into <0, 100>
        number_of_comparisons = 4 # Rewrite should some be added/removed
        worst_case = number_of_comparisons * 100
        score = 100 * (score / worst_case)
        
        return score  # The lower the score, the more similar the two descriptors

