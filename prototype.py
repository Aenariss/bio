import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_gradient_magnitude


def roi(image):
    # Define the region of interest (ROI) in the image
    height, width = image.shape
    roi = np.zeros_like(image, dtype=bool)
    roi[height // 4:3 * height // 4, width // 4:3 * width // 4] = True
    return roi

def detect_finger_vein_mask(image):
    # Apply threshold to separate finger from background
    thresh = roi(image)
    binary = np.zeros_like(image, dtype=bool)
    binary[image > thresh] = True

    # Remove unwanted regions where the finger is not present
    binary = binary.astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=2)

    # Fill holes in the binary mask
    filled_mask = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    return filled_mask

def preprocess_image(image):
    # Generate the vein mask
    vein_mask = detect_finger_vein_mask(image)
    # Apply the mask to isolate the finger region
    masked_image = cv2.bitwise_and(image, image, mask=vein_mask)
    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_image = clahe.apply(masked_image)
    # Gaussian blur to reduce noise
    blurred_image = cv2.GaussianBlur(clahe_image, (5, 5), 0)
    # Unsharp masking to enhance details
    sharp_image = cv2.addWeighted(clahe_image, 1.5, blurred_image, -0.5, 0)

    return vein_mask, masked_image, clahe_image, blurred_image, sharp_image

# Load and process the image
image_path = 'data/002/L_Fore/02.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded. Check the file path or integrity.")

vein_mask, masked_image, clahe_image, blurred_image, sharp_image = preprocess_image(image)
# TODO: add maximum curvature step by step from paper


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

plt.tight_layout()
plt.show()