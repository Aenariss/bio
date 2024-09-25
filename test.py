"""
BIO Project 2024
Author: Vojtech Fiala <xfiala61>
DOESNT WORK
"""

from bob.bio.vein.extractor import MaximumCurvature
import numpy as np
import matplotlib.pyplot as plt
import bob.bio.vein
from skimage import io, color
from skimage.filters import gaussian
from skimage.morphology import binary_opening, disk
from skimage.util import img_as_float
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, morphology
from skimage.util import img_as_float

# Function to preprocess the image
def preprocess_image(image_path):
    # Load the image
    image = io.imread(image_path)

    # Apply Gaussian filter to smooth the image
    smoothed_image = filters.gaussian(image, sigma=1)

    # Apply binary thresholding to create a binary image
    binary_image = smoothed_image > np.mean(smoothed_image)

    # Apply binary opening to reduce noise
    opened_image = morphology.binary_opening(binary_image, morphology.disk(2))

    # Create the finger image and mask
    finger_image = img_as_float(smoothed_image)  # use the smoothed image for curvature
    finger_mask = opened_image.astype(np.float64)  # binary mask of veins

    return finger_image, finger_mask

# Main function to extract features
def extract_features(image_path):
    # Preprocess the image
    finger_image, finger_mask = preprocess_image(image_path)

    # Initialize the Maximum Curvature extractor
    extractor = MaximumCurvature()

    # Extract features
    features = extractor(finger_image, mask=finger_mask)

    return features


# Example usage
if __name__ == "__main__":
    # Path to the image file
    image_path = 'data/001/L_Fore/01.png'

    # Extract features from the image
    features = extract_features(image_path)

    # Display the extracted features
    print("Extracted Features Shape:", features.shape)

    # Optionally visualize the preprocessed image
    plt.imshow(preprocess_image(image_path), cmap='gray')
    plt.title('Preprocessed Vein Image')
    plt.axis('off')
    plt.show()