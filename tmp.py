# Tmp file with NOT FUNCTIONAL max curvature implementation


from src.DataLoader import DataLoader
import numpy as np
import cv2
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

# Dataset Mentioned at https://www.researchgate.net/publication/308128095_A_Review_of_Finger-Vein_Biometrics_Identification_Approaches, found at https://huggingface.co/datasets/luyu0311/MMCBNU_6000
# We use only a part of it 

data = DataLoader().load_images()
test_path = 'data/001/L_Fore/01.bmp'
test_image = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)


#### tmp until it works
# Functions for extracting profiles
def get_vertical_profile(F, x_pos):
    return F[:, x_pos]  # Extract vertical profile from column x_pos

def get_horizontal_profile(F, y_pos):
    return F[y_pos, :]  # Extract horizontal profile from row y_pos

def get_diagonal_profile_1(F):
    return np.diag(np.fliplr(F))  # Extract diagonal profile from bottom-left to top-right

def get_diagonal_profile_2(F):
    return np.diag(F)  # Extract diagonal profile from top-left to bottom-right

# Function for computing curvature
def compute_curvature(P_f):
    try:
        dP_f_dz = np.gradient(P_f)  # Compute first derivative dP_f/dz
        d2P_f_dz2 = np.gradient(dP_f_dz)  # Compute second derivative d^2P_f/dz^2
        curvature = d2P_f_dz2 / (1 + dP_f_dz**2)**(3/2)  # Compute curvature formula
    except:
        return 0
    return curvature

# Find local maxima in concave regions
def find_local_maxima_in_concave_regions(K_z):
    concave_indices = np.where(K_z < 0)[0]  # Find concave regions (K(z) < 0)
    if len(concave_indices) == 0:
        return [], []
    
    concave_mask = np.zeros_like(K_z, dtype=bool)
    concave_mask[concave_indices] = True  # Mask concave regions
    
    all_peaks, _ = find_peaks(K_z)  # Find all local maxima
    concave_peaks = [peak for peak in all_peaks if concave_mask[peak]]  # Filter for concave peaks
    peak_values = K_z[concave_peaks]  # Get the values of local maxima in concave regions
    
    return concave_peaks, peak_values

# Compute score for local maxima
def compute_score_for_local_maxima(K_z, local_maxima):
    scores = []
    for idx in local_maxima:
        left = idx
        right = idx
        
        while left > 0 and K_z[left] > 0:
            left -= 1
        while right < len(K_z) - 1 and K_z[right] > 0:
            right += 1
        
        W_r_i = right - left  # Width of the region with positive curvature
        score = abs(K_z[idx]) * W_r_i  # Score is curvature * width
        scores.append((idx, score))
    
    return scores

# Update the vein plane with scores
def update_vein_plane(V, scores, profile_positions):
    for i, (idx, score) in enumerate(scores):
        try:
            x, y = profile_positions[i]
            V[x, y] += score  # Update the vein plane with the score
        except:
            break
    
    return V

# Main function to analyze profiles in all directions and calculate scores
def analyze_vein_patterns(F):
    rows, cols = F.shape
    V = np.zeros_like(F, dtype=float)  # Vein plane V to store scores
    
    # 1. Vertical Profiles
    for x in range(cols):
        profile = get_vertical_profile(F, x)
        curvature = compute_curvature(profile)
        local_maxima, _ = find_local_maxima_in_concave_regions(curvature)
        scores = compute_score_for_local_maxima(curvature, local_maxima)
        profile_positions = [(y, x) for y, _ in enumerate(local_maxima)]  # (y, x) positions in V
        V = update_vein_plane(V, scores, profile_positions)
    
    # 2. Horizontal Profiles
    for y in range(rows):
        profile = get_horizontal_profile(F, y)
        curvature = compute_curvature(profile)
        local_maxima, _ = find_local_maxima_in_concave_regions(curvature)
        scores = compute_score_for_local_maxima(curvature, local_maxima)
        profile_positions = [(y, x) for x, _ in enumerate(local_maxima)]  # (y, x) positions in V
        V = update_vein_plane(V, scores, profile_positions)
    
    # 3. Diagonal Profiles (bottom-left to top-right)
    for offset in range(-rows + 1, cols):
        diagonal_profile = np.diag(np.fliplr(F), k=offset)  # Diagonal profile
        curvature = compute_curvature(diagonal_profile)
        local_maxima, _ = find_local_maxima_in_concave_regions(curvature)
        scores = compute_score_for_local_maxima(curvature, local_maxima)
        profile_positions = [(i, cols - 1 - (i - offset)) for i in range(len(local_maxima)) if 0 <= (cols - 1 - (i - offset)) < cols]  # (x, y) positions in V
        V = update_vein_plane(V, scores, profile_positions)

    # 4. Diagonal Profiles (top-left to bottom-right)
    for offset in range(-rows + 1, cols):
        diagonal_profile = np.diag(F, k=offset)
        curvature = compute_curvature(diagonal_profile)
        local_maxima, _ = find_local_maxima_in_concave_regions(curvature)
        scores = compute_score_for_local_maxima(curvature, local_maxima)
        profile_positions = [(i, i + offset) for i in range(len(local_maxima)) if 0 <= (i + offset) < cols]  # (x, y) positions in V
        V = update_vein_plane(V, scores, profile_positions)
    
    return V


def compute_Cd1(V, x, y):
    """Calculate C_d1 for horizontal direction."""
    height, width = V.shape
    
    # Ensure x is within the valid range for columns
    if 2 <= x < width - 2:
        left_max = max(V[y, x-1], V[y, x-2])
        right_max = max(V[y, x+1], V[y, x+2])
        return min(left_max + right_max, V[y, x])
    
    # If out of bounds, return the original value or handle accordingly
    return V[y, x]

def compute_Cd2(V, x, y):
    """Calculate C_d2 for vertical direction."""
    height, width = V.shape
    
    # Ensure y is within the valid range for rows
    if 2 <= y < height - 2:
        top_max = max(V[y-1, x], V[y-2, x])
        bottom_max = max(V[y+1, x], V[y+2, x])
        return min(top_max + bottom_max, V[y, x])
    
    # If out of bounds, return the original value or handle accordingly
    return V[y, x]

def compute_Cd3(V, x, y):
    """Calculate C_d3 for diagonal direction (top-left to bottom-right)."""
    height, width = V.shape
    
    # Ensure indices are within valid range for diagonal access
    if 2 <= x < width - 2 and 2 <= y < height - 2:
        diag_tl_br_1 = max(V[y-1, x-1], V[y-2, x-2])
        diag_tl_br_2 = max(V[y+1, x+1], V[y+2, x+2])
        return min(diag_tl_br_1 + diag_tl_br_2, V[y, x])
    
    # If out of bounds, return the original value or handle accordingly
    return V[y, x]

def compute_Cd4(V, x, y):
    """Calculate C_d4 for diagonal direction (bottom-left to top-right)."""
    height, width = V.shape
    
    # Ensure indices are within valid range for diagonal access
    if 2 <= x < width - 2 and 2 <= y < height - 2:
        diag_bl_tr_1 = max(V[y+1, x-1], V[y+2, x-2])
        diag_bl_tr_2 = max(V[y-1, x+1], V[y-2, x+2])
        return min(diag_bl_tr_1 + diag_bl_tr_2, V[y, x])
    
    # If out of bounds, return the original value or handle accordingly
    return V[y, x]

def apply_filtering(V):
    """
    Apply filtering operation to connect vein centers and eliminate noise.
    :param V: 2D numpy array representing the vein plane (scores for vein centers).
    :return: G, the filtered vein pattern image.
    """
    rows, cols = V.shape
    G = np.zeros_like(V)

    # Iterate over every pixel in the vein plane
    for x in range(2, cols - 2):
        for y in range(2, rows - 2):
            # Calculate C_d1, C_d2, C_d3, C_d4 for each pixel
            C_d1 = compute_Cd1(V, x, y)
            C_d2 = compute_Cd2(V, x, y)
            C_d3 = compute_Cd3(V, x, y)
            C_d4 = compute_Cd4(V, x, y)

            # Final value G(x, y) is the maximum of C_d1, C_d2, C_d3, C_d4
            G[y, x] = max(C_d1, C_d2, C_d3, C_d4)

    return G


#prof_0 = get_vertical_profile(test_image, 0) # first column of the image profile - P_f() for x = 0
#curvature_0 = compute_curvature(prof_0) # Calculate curvature at given profile
# If curvature is positive, it's concave (dent), else convex

scores = analyze_vein_patterns(test_image)

veins = apply_filtering(scores)
_, binary_image = cv2.threshold(veins.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Filtered Vein Pattern (G)')
plt.imshow(veins)
plt.subplot(1, 2, 2)
plt.title('Binarized Image')
plt.imshow(binary_image)
plt.show()


# When calling this method, we should already have a mask of the finger available - meaning we run this only on the part of the image containing a finger
