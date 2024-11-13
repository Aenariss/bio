"""
BIO Project 2024
Author: 
    Filip Brna <xbrnaf00>
    Vojtech Fiala <xfiala61>
"""

# Dataset Mentioned at https://www.researchgate.net/publication/308128095_A_Review_of_Finger-Vein_Biometrics_Identification_Approaches, found at https://huggingface.co/datasets/luyu0311/MMCBNU_6000
# We use only a part of it 

from src.Comparator import Comparator
from src.Pipeline import pipeline
from src.FeatureExtractor import FeatureExtractor

import matplotlib.pyplot as plt
import argparse

def show_results(image_path: str):
    """
    Method to visualize the results of the vein extraction and image processing pipeline.
    
    Args:
        image_path (str): Path to the input image for processing and visualization.
    """
    
    # Run the pipeline to obtain the processed images and results
    image, vein_mask, masked_image, clahe_image, blurred_image, sharp_image, result, result_normalized, result_normalized_without_noise = pipeline(image_path, intermediate=True)
    
    import cv2
    cv2.imwrite("output.png", result_normalized_without_noise)
    # Create a 3x3 grid of subplots to display the images
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    # Original Image
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    # Vein Mask
    axs[0, 1].imshow(vein_mask, cmap='gray')
    axs[0, 1].set_title('Finger Vein Mask')
    axs[0, 1].axis('off')

    # Masked Image (image with vein mask applied)
    axs[0, 2].imshow(masked_image, cmap='gray')
    axs[0, 2].set_title('Masked Image')
    axs[0, 2].axis('off')

    # Enhanced Vein Pattern (CLAHE applied)
    axs[1, 0].imshow(clahe_image, cmap='gray')
    axs[1, 0].set_title('Enhanced Vein Pattern')
    axs[1, 0].axis('off')

    # Blurred Image (Gaussian blur applied)
    axs[1, 1].imshow(blurred_image, cmap='gray')
    axs[1, 1].set_title('Blurred Image')
    axs[1, 1].axis('off')

    # Sharpened Image (image sharpened)
    axs[1, 2].imshow(sharp_image, cmap='gray')
    axs[1, 2].set_title('Sharp Image')
    axs[1, 2].axis('off')

    # Maximum Curvature Visualization
    axs[2, 0].imshow(result, cmap='jet')
    axs[2, 0].set_title('Maximum Curvature')
    axs[2, 0].axis('off')

    # Normalized Maximum Curvature
    axs[2, 1].imshow(result_normalized, cmap='gray')
    axs[2, 1].set_title('Normalized Maximum Curvature')
    axs[2, 1].axis('off')

    # Maximum Curvature without Noise
    axs[2, 2].imshow(result_normalized_without_noise, cmap='gray')
    axs[2, 2].set_title('Without Noise')
    axs[2, 2].axis('off')

    # Adjust layout to prevent overlapping
    plt.tight_layout()

    # Display the results
    plt.show()


def compare_2_images(img1: str, img2: str):
    """
    Compares two finger vein images by extracting and comparing their feature descriptors.
    
    Args:
        img1 (str): Path to the first image for comparison.
        img2 (str): Path to the second image for comparison.
    """
    # Process the images and obtain vein masks
    img1, mask1 = pipeline(img1)
    img2, mask2 = pipeline(img2)

    # Initialize comparator
    comparator = Comparator()

    # Align the second image with the first using ECC (Enhanced Correlation Coefficient)
    img2 = comparator.align_images(img1, img2, mask1, mask2)

    # Extract feature descriptors from both images
    features1 = FeatureExtractor(img1)
    descriptor1 = features1.get_features()

    features2 = FeatureExtractor(img2)
    descriptor2 = features2.get_features()

    # Calculate and print the similarity score
    score = comparator.compare_descriptors(descriptor1, descriptor2)
    if score < 60:
        print("Match!")
    else:
        print("Non-match!")
    print(f"Similarity Score: {score}")


if __name__ == "__main__":
    # Command-line argument parsing for image comparison or result visualization
    parser = argparse.ArgumentParser(description='Finger vein extraction and comparison. The default (no argument) visualizes the image processing pipeline results.')
    parser.add_argument('-ip', '--image-path', type=str, help='Path to the finger image', required=True)
    parser.add_argument('-cw', '--compare-with', type=str, help='Path to the finger image to compare with')
    parser.add_argument('-ca', '--compare-all', action='store_true', help='Compare the given image with all images in the dataset')
    args = parser.parse_args()

    # Error handling for conflicting arguments
    if args.compare_with and args.compare_all:
        parser.error("Cannot use both --compare-with and --compare-all arguments simultaneously.")
        exit(1)

    # Run the appropriate function based on the arguments
    if args.compare_with:
        compare_2_images(args.image_path, args.compare_with)
    elif args.compare_all:
        Comparator().compare_all(args.image_path)
    else:
        show_results(args.image_path)
