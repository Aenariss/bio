"""
BIO Project 2024

This module contains a class to compare results 
Author: 
    Vojtech Fiala <xfiala61>
    ChatGPT
    Gemini
    Claude
"""

import numpy as np
import cv2

from src.DataLoader import DataLoader
from src.Pipeline import pipeline
from src.FeatureExtractor import FeatureExtractor

class Comparator:
    def __init__(self, threshold: int = 60):
        """
        Initializes the ImageComparator with a similarity threshold.
        
        Args:
            threshold (int): Threshold value for similarity comparison.
                             Images with a similarity score below this threshold are considered a match.
        """
        self.threshold = threshold

    def align_images(self, image1: np.ndarray, image2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray) -> np.ndarray:
        """
        Aligns `image2` to `image1` using the Enhanced Correlation Coefficient (ECC) method.
        
        Args:
            image1 (np.ndarray): The reference image to which `image2` will be aligned.
            image2 (np.ndarray): The image to align with `image1`.
            mask1 (np.ndarray): Binary mask of `image1`, used in alignment.
            mask2 (np.ndarray): Binary mask of `image2`, used in alignment.
        
        Returns:
            np.ndarray: `image2` aligned to `image1` using an affine transformation.
        """
        # Define motion model allowing translation and rotation
        warp_mode = cv2.MOTION_EUCLIDEAN

        # Define termination criteria for the ECC algorithm: maximum iterations or epsilon for convergence
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 1000, 1e-8)
        
        # Initialize the warp matrix for affine transformation
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        
        # Compute the warp matrix using the ECC algorithm
        _, warp_matrix = cv2.findTransformECC(
            mask1, 
            mask2,
            warp_matrix, 
            warp_mode,
            criteria,
            inputMask=None,
            gaussFiltSize=1
        )
        
        # Apply the affine transformation to align image2 with image1
        img2_binary_aligned = cv2.warpAffine(
            image2, 
            warp_matrix, 
            (image1.shape[1], image1.shape[0]),
            flags=cv2.INTER_NEAREST + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_CONSTANT
        )

        return img2_binary_aligned

    def __local_histogram_comparison(self, h1: list, h2: list) -> float:
        """
        Compares two lists of histograms using correlation to evaluate similarity.
        
        Args:
            h1 (list): List of histograms from image patches of the first image.
            h2 (list): List of histograms from image patches of the second image.
        
        Returns:
            float: Normalized similarity score, where a lower score indicates higher similarity.
        
        Raises:
            RuntimeError: If the input histograms do not match in length, indicating mismatched images.
        """
        # Ensure both histogram lists have the same length for valid comparison
        n_of_hists = len(h1)
        if n_of_hists != len(h2):
            raise RuntimeError("Histograms of different sizes indicate invalid images were used!")

        # List to store similarity scores for each patch
        similarity_scores = []

        for i in range(n_of_hists):
            hist1 = h1[i]
            hist2 = h2[i]

            # Normalize both histograms
            hist1 = cv2.normalize(hist1, hist1).flatten()
            hist2 = cv2.normalize(hist2, hist2).flatten()

            # Compute correlation between histograms to evaluate similarity
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            # Add the similarity score for this patch to the list
            similarity_scores.append(correlation)

        # Compute the average similarity score across all patches
        avg_similarity = np.mean(similarity_scores)

        # Normalize similarity score to range [0, 100000], where lower values indicate higher similarity
        normalized_score = 100000 * (1 - avg_similarity)

        return normalized_score

    def compare_all(self, compared_img_path: str) -> list:
        """
        Compare multiple images in a dataset against a given reference image.
        
        Args:
            compared_img_path (str): Path to the reference image against which all other images will be compared.
        
        Returns:
            list: A list of similarity scores where each score represents the similarity between the reference image
                  and each image in the dataset. Lower scores indicate better matches.
        """
        results = []
        
        # Initialize DataLoader with the reference image path
        loader = DataLoader(original_finger=compared_img_path)
        
        # Load images for comparison
        data = loader.load_images()
        
        # Extract data identifying the original image owner
        original_id_person, original_finger, original_id_finger = loader.get_original_image_data()
        
        # Extract vein pattern and mask from the reference image
        original_veins, original_mask = pipeline(compared_img_path)
        
        # Extract features from the reference image
        original_featuresExtractor = FeatureExtractor(original_veins)
        original_features = original_featuresExtractor.get_features()

        print(f"Starting comparison of person {original_id_person} finger {original_finger} photo {original_id_finger} with all dataset entries...")

        false_negatives = 0
        false_positives = 0
        true_matches = 0
        true_nonmatches = 0

        same_matches = 0
        different_matches = 0

        total_finger_count = 0

        break_flag = False
        
        for person in data:
            if break_flag:
                break
            print(f"Comparing reference person with person {person}...")
            for finger in data[person]:
                if break_flag:
                    break
                print(f"Comparing reference finger {original_finger} from person {original_id_person} with finger {finger} of person {person}...")
                for photo in data[person][finger]:
                    # Every 30 finger images, change the original to have True Positive or False Negative that make sense, not just from 10 samples
                    if (total_finger_count % 30 == 0):
                        # Extract data identifying the original image owner
                        original_id_person, original_finger, original_id_finger = person, finger, photo

                        compared_img_path = original_id_finger
                        
                        # Extract vein pattern and mask from the reference image
                        original_veins, original_mask = pipeline(compared_img_path)
                        
                        # Extract features from the reference image
                        original_featuresExtractor = FeatureExtractor(original_veins)
                        original_features = original_featuresExtractor.get_features()
                    # Extract vein pattern and mask from the current photo
                    current_veins, current_mask = pipeline(photo)
                    
                    # Align current image to the reference image
                    current_veins = self.align_images(original_veins, current_veins, original_mask, current_mask)

                    # Extract features from the aligned image
                    cmp_features = FeatureExtractor(current_veins)
                    cmp_descriptor = cmp_features.get_features()
                    
                    # Compare features and calculate similarity score
                    result_score = self.compare_descriptors(original_features, cmp_descriptor)

                    if result_score < self.threshold:
                        if person == original_id_person and finger == original_finger:
                            true_matches += 1
                            same_matches += 1
                        else:
                            false_positives += 1
                            different_matches += 1
                        print("Match!", end=' ')
                    else:
                        if person == original_id_person and finger == original_finger:
                            false_negatives += 1
                            same_matches += 1
                        else:
                            true_nonmatches += 1
                            different_matches += 1
                        print("Non-match!", end=' ')

                    print(f"Similarity score: {result_score}")
                

                    results.append(result_score)

                    total_finger_count += 1

                    #if (total_finger_count == 30):
                    #    break_flag = True
                    
        print(f"False Match Rate (False positive): {false_positives / different_matches}")
        print(f"False Non-Match Rate (False negative): {false_negatives / same_matches}")
        print(f"True Match Rate (True positive): {true_matches / same_matches}")
        print(f"True Non-Match Rate (True Negative): {true_nonmatches / different_matches}")
        return results

    def __compare_bifurcations(self, img, b1: list, b2: list, endpoints: bool = False, distance_threshold: int = 20) -> float:
        """
        Compares two sets of bifurcations or endpoints, calculating a similarity score based on spatial and
        structural proximity.
        
        Args:
            b1 (list): List of bifurcation or endpoint coordinates and properties for the reference image.
            b2 (list): List of bifurcation or endpoint coordinates and properties for the comparison image.
            endpoints (bool): Whether the points represent endpoints (True) or bifurcations (False).
            distance_threshold (int): Maximum distance between points in b1 and b2 to be considered a match.
        
        Returns:
            float: A normalized similarity score, where lower values indicate a higher similarity.
        """
        # Convert bifurcation lists to numpy arrays for easier manipulation
        b1 = np.array(b1)
        b2 = np.array(b2)

        # Return maximum penalty if either list is empty, indicating no features
        if len(b1) == 0 or len(b2) == 0:
            return 100

        # Extract x, y coordinates from both sets of bifurcations
        coords1 = np.array(b1[:, :2], dtype=int)
        coords2 = np.array(b2[:, :2], dtype=int)

        '''
        # Create a color image (3 channels) with the same size as the grayscale image, initialized to black
        colored_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        for coord in coords1:
            y, x = coord  # Unpack the coordinates
            # Draw a circle at each coordinate
            cv2.circle(colored_image , (x, y), radius=5, color=(255, 0, 0), thickness=1)
        colored_image = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 1, colored_image, 1, 0)
        cv2.imwrite("output.png", colored_image)
        '''
        
        
        # Calculate pairwise Euclidean distances between all points in coords1 and coords2
        diff = coords1[:, None, :] - coords2[None, :, :]
        distances = np.sqrt(np.sum(diff**2, axis=2))
        
        # Initialize structures for tracking matched points and penalties
        matches = []
        matched_indices_list2 = set()
        
        # Define maximum penalty, adjusted based on whether points are endpoints or bifurcations
        maximum_penalty = distance_threshold if endpoints else distance_threshold + 15

        for i in range(distances.shape[0]):
            # Identify valid matches within the threshold for the current point in b1
            within_threshold = distances[i] <= distance_threshold
            if np.any(within_threshold):
                # Select the closest match within threshold
                min_dist_index = np.argmin(np.where(within_threshold, distances[i], np.inf))
                min_distance = distances[i, min_dist_index]
                
                # Only consider this match if the point in b2 hasn't already been matched
                if min_dist_index not in matched_indices_list2:
                    matched_indices_list2.add(min_dist_index)

                    if not endpoints:
                        # Bifurcation: compare shape and direction properties for additional penalties
                        shape1, dir1 = b1[i, 2], b1[i, 3]
                        shape2, dir2 = b2[min_dist_index, 2], b2[min_dist_index, 3]
                        
                        shape_match = shape1 == shape2
                        direction_match = dir1 == dir2
                        
                        penalty = min_distance + 10 * int(not shape_match) + 5 * int(not direction_match)
                        matches.append(penalty)
                    else:
                        # Endpoints: only distance penalty is applied
                        matches.append(min_distance)
            else:
                # No valid match, apply maximum penalty
                matches.append(maximum_penalty)
                
        # Add penalties for any unmatched points in coords2
        additional_penalty = sum([maximum_penalty for j in range(len(coords2)) if j not in matched_indices_list2])

        # Calculate the total penalty and normalize it to a [0, 100] scale
        total_penalty = sum(matches) + additional_penalty
        maximum_possible_penalty = (len(coords1) + len(coords2)) * maximum_penalty
        normalized_penalty = 100 * (total_penalty / maximum_possible_penalty)

        return normalized_penalty

    def __compare_endpoints(self, skelet, e1: list, e2: list) -> float:
        """
        Compares the endpoints of two vein structures using a Euclidean distance-based comparison.
        
        Args:
            e1 (list): List of endpoints for the reference image.
            e2 (list): List of endpoints for the comparison image.
        
        Returns:
            float: A similarity score, where lower values indicate more similarity.
        """
        return self.__compare_bifurcations(skelet, e1, e2, endpoints=True, distance_threshold=50)

    def __compare_overlap(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """
        Computes the overlap percentage between two aligned binary images.
        
        Args:
            img1 (np.ndarray): Binary image (reference) after alignment.
            img2 (np.ndarray): Binary image (comparison) after alignment.
        
        Returns:
            float: The difference in percentage overlap between the two images. Lower values indicate higher similarity.
        """
        # Convert images to binary format for overlap calculation
        img1_binary = (img1 > 0).astype(np.uint8)
        img2_binary = (img2 > 0).astype(np.uint8)

        # Calculate overlap (intersection) and union areas
        overlap = img1_binary * img2_binary
        overlap_count = np.sum(overlap)
        union_count = np.sum(img1_binary) + np.sum(img2_binary) - overlap_count
        
        # Calculate overlap percentage and return the difference percentage (inverse similarity)
        overlap_percentage = (overlap_count / union_count) * 100 if union_count != 0 else 0
        return 100 - overlap_percentage  # How much they differ in percentage

    def compare_descriptors(self, features1: dict, features2: dict) -> float:
        """
        Compares two feature descriptors and calculates an overall similarity score.
        
        Args:
            features1 (dict): Feature descriptor of the reference image, containing bifurcations, endpoints,
                              local histograms, max curvature, and skeleton information.
            features2 (dict): Feature descriptor of the comparison image, containing bifurcations, endpoints,
                              local histograms, max curvature, and skeleton information.
        
        Returns:
            float: A similarity score normalized to the range [0, 100], where lower values indicate higher similarity.
        """
        score = 0

        # Set weighting factor for more important features
        more_important_increase = 1.5

        # Compare bifurcations, contributing to similarity score
        score += self.__compare_bifurcations(features1['skeleton'], features1['bifurcations'], features2['bifurcations'])
        
        # Compare local histograms across regions
        score += self.__local_histogram_comparison(features1['localHistograms'], features2['localHistograms'])

        # Compare image overlap based on maximum curvature for structural similarity
        score += more_important_increase * self.__compare_overlap(features1['maxCurvature'], features2['maxCurvature'])

        # Compare endpoints, adding to the similarity score
        score += self.__compare_endpoints(features1['skeleton'], features1['endpoints'], features2['endpoints'])

        # Normalize score to [0, 100] scale; lower scores indicate higher similarity
        number_of_comparisons = 4  # Update if features are added or removed
        not_standard = 1
        worst_case = more_important_increase * not_standard * 100 + (number_of_comparisons - not_standard) * 100
        score = 100 * (score / worst_case)
        
        return score  # Lower score indicates greater similarity
