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

def show_results(image_path):
    """
    Method to visualize the results
    """
    
    image, vein_mask, masked_image, clahe_image, blurred_image, sharp_image, result, result_normalized, result_normalized_without_noise = pipeline(image_path, intermediate=True)

    descriptor = FeatureExtractor(result_normalized_without_noise).create_descriptor()
    #print(descriptor)

    # Plotting
    fig, axs = plt.subplots(3, 3, figsize=(10, 10))

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Original Image')
    axs[0, 0].axis('off')

    axs[0, 1].imshow(vein_mask, cmap='gray')
    axs[0, 1].set_title('Finger Vein Mask')
    axs[0, 1].axis('off')

    axs[0, 2].imshow(masked_image, cmap='gray')
    axs[0, 2].set_title('Masked Image')
    axs[0, 2].axis('off')

    axs[1, 0].imshow(clahe_image, cmap='gray')
    axs[1, 0].set_title('Enhanced Vein Pattern')
    axs[1, 0].axis('off')

    axs[1, 1].imshow(blurred_image, cmap='gray')
    axs[1, 1].set_title('Blurred Image')
    axs[1, 1].axis('off')

    axs[1, 2].imshow(sharp_image, cmap='gray')
    axs[1, 2].set_title('Sharp Image')
    axs[1, 2].axis('off')

    axs[2, 0].imshow(result, cmap='jet')
    axs[2, 0].set_title('Maximum Curvature')
    axs[2, 0].axis('off')

    axs[2, 1].imshow(result_normalized, cmap='gray')
    axs[2, 1].set_title('Normalized Maximum Curvature')
    axs[2, 1].axis('off')

    axs[2, 2].imshow(result_normalized_without_noise, cmap='gray')
    axs[2, 2].set_title('Without Noise')
    axs[2, 2].axis('off')

    #plt.savefig('tmp_result.png')

    plt.tight_layout()
    plt.show()

def compare_2_images(img1, img2):
    img1 = pipeline(img1)
    img2 = pipeline(img2)

    descriptor1 = FeatureExtractor(img1).create_descriptor()
    descriptor2 = FeatureExtractor(img2).create_descriptor()

    #score = Comparator(threshold=14).compare(img1, img2)

    # The lower score the more similar they are
    score = Comparator().compare_descriptors(descriptor1, descriptor2)
    print(score)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Finger vein extraction and comparison, default(no argument) is to visualize the pipeline of the image')
    parser.add_argument('-ip', '--image-path', type=str, help='Path to the finger image', required=True)
    parser.add_argument('-cw', '--compare-with', type=str, help='Path to the finger image to compare with')
    parser.add_argument('-ca', '--compare-all', action='store_true', help='Compare all images in the dataset')
    args = parser.parse_args()

    if args.compare_with and args.compare_all:
        parser.error("Cannot use both compare_with and compare_all arguments")
        exit(1)

    if args.compare_with:
        compare_2_images(args.image_path, args.compare_with)
    elif args.compare_all:
        Comparator().compare_all(args.image_path)
    else:
        show_results(args.image_path)