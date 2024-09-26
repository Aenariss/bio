import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage

def roi(image):
    # Define the region of interest (ROI) in the image
    height, width = image.shape
    roi = np.zeros_like(image, dtype=bool)
    roi[height // 4:3 * height // 4, width // 4:3 * width // 4] = True
    return roi

def detect_finger_vein_mask(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply threshold to separate finger from background
    thresh = roi(image)
    binary = np.zeros_like(image, dtype=bool)
    binary[image > thresh] = True

    # remove unwanted regions where the finger is not present
    binary = binary.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # Fill holes in the binary mask
    filled_mask = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    return filled_mask

def plot_results(original_image, mask):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    ax1.imshow(original_image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')
    
    ax2.imshow(mask, cmap='gray')
    ax2.set_title('Finger Vein Mask')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
image_path = 'data/002/L_Fore/01.bmp'
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
mask = detect_finger_vein_mask(image_path)
plot_results(original_image, mask)