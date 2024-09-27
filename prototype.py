import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_gradient_magnitude
import math

# ---- Preprocessing Functions ----

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


# ---- Maximum Curvature Functions ----

def meshgrid(xgv, ygv):
    X, Y = np.meshgrid(xgv, ygv)
    return X, Y

def max_curvature(src, mask, sigma=8):
    src = src.astype(np.float32) / 255.0
    sigma2 = sigma ** 2
    sigma4 = sigma ** 4

    winsize = int(np.ceil(4 * sigma))
    X, Y = meshgrid(np.arange(-winsize, winsize + 1), np.arange(-winsize, winsize + 1))

    X2 = np.power(X, 2)
    Y2 = np.power(Y, 2)
    X2Y2 = X2 + Y2

    h = (1 / (2 * np.pi * sigma2)) * np.exp(-X2Y2 / (2 * sigma2))
    hx = -(X / sigma2) * h
    hxx = ((X2 - sigma2) / sigma4) * h
    hy = hx.T
    hyy = hxx.T
    hxy = (X * Y / sigma4) * h

    fx = -cv2.filter2D(src, -1, hx)
    fy = cv2.filter2D(src, -1, hy)
    fxx = cv2.filter2D(src, -1, hxx)
    fyy = cv2.filter2D(src, -1, hyy)
    fxy = -cv2.filter2D(src, -1, hxy)

    f1 = 0.5 * np.sqrt(2.0) * (fx + fy)
    f2 = 0.5 * np.sqrt(2.0) * (fx - fy)
    f11 = 0.5 * fxx + fxy + 0.5 * fyy
    f22 = 0.5 * fxx - fxy + 0.5 * fyy

    k1 = np.zeros_like(src)
    k2 = np.zeros_like(src)
    k3 = np.zeros_like(src)
    k4 = np.zeros_like(src)

    for y in range(src.shape[0]):
        for x in range(src.shape[1]):
            if mask[y, x] > 0:
                k1[y, x] = fxx[y, x] / np.power(1 + fx[y, x]**2, 1.5)
                k2[y, x] = fyy[y, x] / np.power(1 + fy[y, x]**2, 1.5)
                k3[y, x] = f11[y, x] / np.power(1 + f1[y, x]**2, 1.5)
                k4[y, x] = f22[y, x] / np.power(1 + f2[y, x]**2, 1.5)

    return np.maximum(np.maximum(k1, k2), np.maximum(k3, k4))

# --- Main ---

# Load and process the image
image_path = 'data/003/L_Fore/09.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded. Check the file path or integrity.")

# Preprocess the image
vein_mask, masked_image, clahe_image, blurred_image, sharp_image = preprocess_image(image)
# do the max curvature
result = max_curvature(sharp_image, vein_mask, sigma=8)


# Normalize the result to binary with lower 
result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))
#result_normalized = (result_normalized > 0.3).astype(np.uint8) * 255

# Assuming `result_normalized` is your image in the range [0, 1]
result_scaled = (result_normalized * 255).astype(np.uint8)

# Apply adaptive thresholding
result_normalized = cv2.adaptiveThreshold(result_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 1)

# remove white noise
result_normalized_without_noise = cv2.morphologyEx(result_normalized, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)

# Load and process the image
image_path = 'data/001/L_Fore/05.bmp'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded. Check the file path or integrity.")

# Preprocess the image
vein_mask, masked_image, clahe_image, blurred_image, sharp_image = preprocess_image(image)
# do the max curvature
result = max_curvature(sharp_image, vein_mask, sigma=8)


# Normalize the result to binary with lower 
result_normalized = (result - np.min(result)) / (np.max(result) - np.min(result))
#result_normalized = (result_normalized > 0.3).astype(np.uint8) * 255

# Assuming `result_normalized` is your image in the range [0, 1]
result_scaled = (result_normalized * 255).astype(np.uint8)

# Apply adaptive thresholding
result_normalized = cv2.adaptiveThreshold(result_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 81, 1)

# remove white noise
result_normalized_without_noise_2 = cv2.morphologyEx(result_normalized, cv2.MORPH_OPEN, np.ones((5,5), np.uint8), iterations=2)


def compute_phase_correlation(img1, img2):
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

    return max_corr_value

# Compute the Miura match score
score = compute_phase_correlation(result_normalized_without_noise, result_normalized_without_noise_2)
score = int(1000 * score)

if score is not None:
    print(f"Match! Score:" if score >= 14 else "Not match! Score:", score)
else:
    print("Error in computing the Miura match score.")




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

plt.tight_layout()
plt.show()