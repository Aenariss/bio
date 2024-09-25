import cv2
import numpy as np
import matplotlib.pyplot as plt

def enhance_contrast(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)

def calculate_curvature(profile):
    # Approximate the second derivative
    second_derivative = np.gradient(np.gradient(profile))
    curvature = second_derivative / (1 + np.gradient(profile)**2)**1.5
    return curvature

def extract_vein_pattern(image):
    height, width = image.shape
    vein_map = np.zeros_like(image, dtype=float)

    for direction in ['horizontal', 'vertical']:
        for i in range(height if direction == 'horizontal' else width):
            if direction == 'horizontal':
                profile = image[i, :]
            else:
                profile = image[:, i]
            
            curvature = calculate_curvature(profile)
            
            # Find local maxima of positive curvature
            positive_curvature = np.maximum(curvature, 0)
            local_maxima = (positive_curvature > np.roll(positive_curvature, 1)) & \
                           (positive_curvature > np.roll(positive_curvature, -1))
            
            # Assign scores based on curvature and width
            scores = np.zeros_like(profile, dtype=float)
            for j in np.where(local_maxima)[0]:
                # Estimate width of the concave region
                left = j
                while left > 0 and curvature[left] > 0:
                    left -= 1
                right = j
                while right < len(curvature) - 1 and curvature[right] > 0:
                    right += 1
                width = right - left
                
                scores[j] = positive_curvature[j] * width
            
            if direction == 'horizontal':
                vein_map[i, :] += scores
            else:
                vein_map[:, i] += scores

    # Normalize and threshold
    vein_map = (vein_map - np.min(vein_map)) / (np.max(vein_map) - np.min(vein_map))
    vein_mask = (vein_map > 0.2).astype(np.uint8) * 255

    return vein_mask

# Load and process the image
image_path = 'data/001/L_Fore/01.png'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise ValueError("Image not loaded. Check the file path or integrity.")

enhanced_image = enhance_contrast(image)
blurred_image = cv2.GaussianBlur(enhanced_image, (5, 5), 0)
vein_mask = extract_vein_pattern(blurred_image)

# Plotting
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('Original Image')
axs[0, 0].axis('off')

axs[0, 1].imshow(enhanced_image, cmap='gray')
axs[0, 1].set_title('Enhanced Image')
axs[0, 1].axis('off')

axs[1, 0].imshow(blurred_image, cmap='gray')
axs[1, 0].set_title('Blurred Image')
axs[1, 0].axis('off')

axs[1, 1].imshow(vein_mask, cmap='gray')
axs[1, 1].set_title('Vein Mask (Final Output)')
axs[1, 1].axis('off')

# Adding arrows
arrow_props = dict(facecolor='black', arrowstyle='->', lw=2)
axs[0, 0].annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_props, annotation_clip=False)
axs[0, 1].annotate('', xy=(0.5, -0.15), xytext=(0.5, 0.05), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_props, annotation_clip=False)
axs[1, 0].annotate('', xy=(1.05, 0.5), xytext=(0.95, 0.5), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_props, annotation_clip=False)

plt.tight_layout()
plt.show()