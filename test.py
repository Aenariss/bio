"""
BIO Project 2024

File contains tests conducted using the reference bob.bio.vein library.
ChatGPT doesnt know this library and its documentation is absolutely horrible. 
Even though the actual work is done in only 4 lines, the amount of time it took to make them work is unspeakable
Author: Vojtech Fiala <xfiala61> + ChatGPT
"""

# Import necessary libraries
import bob.bio.vein
import bob.bio.vein.extractor
import bob.bio.vein.preprocessor
import bob.io.image
import matplotlib.pyplot as plt
import cv2
from src.Preprocessor import Preprocessor
from src.MaxCurvature import MaxCurvature

from src.Pipeline import pipeline
import numpy as np

def bob_preprocess(img_path):
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    mask = Preprocessor().detect_finger_vein_mask(image)

    # Preprocess the image to get it ready for feature extraction
    preprocessor = bob.bio.vein.preprocessor.HuangNormalization()
    preprocessed_image = preprocessor(image, mask)
    return preprocessed_image

def bob_maxCurve(img):
    # Load the Maximum Curvature extractor from the bob.bio.vein library
    extractor = bob.bio.vein.extractor.MaximumCurvature()

    # Extract the vein pattern using the Maximum Curvature technique
    vein_pattern = extractor(img)
    return vein_pattern


def implemented_maxCurve(img):
    # Our implementation of the maximum curvature method

    return pipeline(img)


def visually_compare_2_figs(fig1, fig2):
    # Display the original image and extracted vein pattern
    plt.figure(figsize=(10, 5))

    # Show original vein image
    plt.subplot(1, 2, 1)
    plt.imshow(fig1, cmap='gray')
    plt.title('BoB reference')
    plt.axis('off')

    # Show extracted vein pattern
    plt.subplot(1, 2, 2)
    plt.imshow(fig2, cmap='gray')
    plt.title('Our implemented method')
    plt.axis('off')

    # Show the plots
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load the vein image (replace with the actual path to your image)
    image_path = "data/001/L_Fore/01.bmp"

    preprocessed = bob_preprocess(image_path)

    bob_implementation = bob_maxCurve(preprocessed)
    our_implementation = implemented_maxCurve(preprocessed[0])
    visually_compare_2_figs(bob_implementation, our_implementation)
