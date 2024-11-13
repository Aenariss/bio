"""
BIO Project 2024

This module contains a class to extract features using optimized max curvature method 
Author: 
    Filip Brna <xbrnaf00>
    ChatGPT
"""

import numpy as np
import cv2

# Define the MaxCurvature class
class MaxCurvature:
    def __init__(self):
        # Constructor method, but nothing is initialized here
        pass

    def meshgrid(self, xgv: np.ndarray, ygv: np.ndarray) -> tuple:
        """
        Create a 2D grid (meshgrid) from given x and y grid vectors (1D arrays).
        
        xgv: np.ndarray
            A 1D array of x-coordinate values, which will be used to create the grid.
            
        ygv: np.ndarray
            A 1D array of y-coordinate values, which will be used to create the grid.
        
        Returns:
            X, Y: tuple
                Two 2D arrays (grids) representing the meshgrid created from xgv and ygv.
        """
        # Create two 2D arrays (grids) from x and y grid vectors using np.meshgrid
        X, Y = np.meshgrid(xgv, ygv)
        
        # Return the generated meshgrid
        return X, Y

    def max_curvature(self, src: np.ndarray, mask: np.ndarray, sigma: int = 8) -> np.ndarray:
        """
        Compute the maximum curvature of the given image using Gaussian derivatives.
        
        src: np.ndarray
            The input grayscale image to compute curvature on.
        
        mask: np.ndarray
            A binary mask that defines the region of interest (ROI) where curvature will be computed.
        
        sigma: int, optional (default=8)
            The standard deviation for the Gaussian kernel used in curvature computation.
        
        Returns:
            Vt: np.ndarray
                A 2D array (same size as the input image) representing the curvature scores at each pixel.
        """
        
        # Normalize the source image to the range [0, 1] by dividing by 255.0
        src = src.astype(np.float32) / 255.0

        # Precompute sigma^2 and sigma^4 for use in Gaussian derivatives
        sigma2 = sigma ** 2
        sigma4 = sigma ** 4

        # Determine the window size based on sigma (to limit the kernel size for convolution)
        winsize = int(np.ceil(4 * sigma))

        # Generate a meshgrid of coordinates in the range of [-winsize, winsize] for both X and Y
        X, Y = self.meshgrid(np.arange(-winsize, winsize + 1), np.arange(-winsize, winsize + 1))

        # Compute X^2, Y^2, and their sum X^2 + Y^2 for Gaussian kernel calculations
        X2 = np.power(X, 2)
        Y2 = np.power(Y, 2)
        X2Y2 = X2 + Y2

        # STEP 1-1: Compute Gaussian kernel and its derivatives
        h = (1 / (2 * np.pi * sigma2)) * np.exp(-X2Y2 / (2 * sigma2))  # Gaussian function
        hx = -(X / sigma2) * h  # First derivative of the Gaussian in the x direction
        hxx = ((X2 - sigma2) / sigma4) * h  # Second derivative of the Gaussian in the x direction
        hy = hx.T  # First derivative of Gaussian in the y direction (transpose of hx)
        hyy = hxx.T  # Second derivative of Gaussian in the y direction (transpose of hxx)
        hxy = (X * Y / sigma4) * h  # Mixed second derivative of the Gaussian (xy direction)

        # STEP 1-2: Apply Gaussian derivatives to the source image using 2D convolution
        # These represent the first and second derivatives of the image in both x and y directions
        fx = -cv2.filter2D(src, -1, hx)  # First derivative in the x direction
        fy = cv2.filter2D(src, -1, hy)  # First derivative in the y direction
        fxx = cv2.filter2D(src, -1, hxx)  # Second derivative in the x direction
        fyy = cv2.filter2D(src, -1, hyy)  # Second derivative in the y direction
        fxy = -cv2.filter2D(src, -1, hxy)  # Mixed second derivative (xy direction)

        # STEP 1-3: Compute diagonal derivatives using fx and fy
        f1 = 0.5 * np.sqrt(2.0) * (fx + fy)  # First diagonal derivative
        f2 = 0.5 * np.sqrt(2.0) * (fx - fy)  # Second diagonal derivative

        # Compute diagonal second derivatives
        f11 = 0.5 * fxx + fxy + 0.5 * fyy  # Second derivative along one diagonal
        f22 = 0.5 * fxx - fxy + 0.5 * fyy  # Second derivative along the other diagonal

        # STEP 1-4: Initialize four curvature matrices, k1, k2, k3, k4, with zeros
        k1 = np.zeros_like(src)
        k2 = np.zeros_like(src)
        k3 = np.zeros_like(src)
        k4 = np.zeros_like(src)

        # Create mask for valid pixels (where mask > 0)
        valid_mask = mask > 0

        # Precalculate denominators for curvature formulas
        denom1 = np.power(1 + fx**2, 1.5)
        denom2 = np.power(1 + fy**2, 1.5)
        denom3 = np.power(1 + f1**2, 1.5)
        denom4 = np.power(1 + f2**2, 1.5)

        # Compute curvatures at each pixel of the region of interest (ROI)
        k1[valid_mask] = fxx[valid_mask] / denom1[valid_mask]
        k2[valid_mask] = fyy[valid_mask] / denom2[valid_mask]
        k3[valid_mask] = f11[valid_mask] / denom3[valid_mask]
        k4[valid_mask] = f22[valid_mask] / denom4[valid_mask]

        # Step 2: Process horizontal, vertical, and diagonal curvatures to find vein connections
        # Initialize the final score matrix for vein connections (Vt)
        img_h, img_w = src.shape[:2]
        Vt = np.zeros_like(src)

        # Horizontal direction processing
        # Using boolean masks and numpy operations to find regions with positive curvature
        for y in range(img_h):
            positive_mask = k1[y, :] > 0  # Find positions where k1 (curvature) > 0
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)  # Detect boundaries
            starts = np.where(changes == 1)[0]  # Find start of positive regions
            ends = np.where(changes == -1)[0]  # Find end of positive regions

            # Process each continuous positive curvature region
            for start, end in zip(starts, ends):
                region = k1[y, start:end]  # Extract the region
                Wr = len(region)  # Width of the region
                max_val = np.max(region)  # Max curvature in the region
                pos_max = start + np.argmax(region)  # Position of the max value

                # Update Vt (curvature score) at the maximum position
                Vt[y, pos_max] += max_val * Wr

        # Vertical direction processing
        for x in range(img_w):
            positive_mask = k2[:, x] > 0  # Find positions where k2 (curvature) > 0
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)  # Detect boundaries
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]

            # Process each continuous positive curvature region
            for start, end in zip(starts, ends):
                region = k2[start:end, x]
                Wr = len(region)
                max_val = np.max(region)
                pos_max = start + np.argmax(region)

                # Update Vt at the maximum position
                Vt[pos_max, x] += max_val * Wr

        # Diagonal (\) direction processing
        for start in range(img_h + img_w - 1):
            # Get starting position
            if start < img_w:
                x_start, y_start = start, 0
            else:
                x_start, y_start = 0, start - img_w + 1
                
            # Get diagonal values
            x_indices = []
            y_indices = []
            x, y = x_start, y_start
            while x < img_w and y < img_h:
                x_indices.append(x)
                y_indices.append(y)
                x += 1
                y += 1
                
            if not x_indices:  # Skip if diagonal is empty
                continue
                
            # Convert to numpy arrays for vectorized operations
            x_indices = np.array(x_indices)
            y_indices = np.array(y_indices)
            diag_values = k3[y_indices, x_indices]
            
            # Find positions where k3 > 0
            positive_mask = diag_values > 0
            
            # Find the boundaries of continuous positive regions
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # Process each continuous region
            for s, e in zip(starts, ends):
                # Get region coordinates
                region_x = x_indices[s:e]
                region_y = y_indices[s:e]
                region = diag_values[s:e]
                Wr = len(region)
                
                if Wr > 0:
                    # Get max value and its position
                    max_val = np.max(region)
                    pos_max = np.argmax(region)
                    
                    # Calculate actual position in original array
                    max_x = region_x[pos_max]
                    max_y = region_y[pos_max]
                    
                    # Update Vt
                    Vt[max_y, max_x] += max_val * Wr

        # Diagonal (/) direction processing
        for start in range(img_h + img_w - 1):
            # Get starting position
            if start < img_w:
                x_start, y_start = start, img_h - 1
            else:
                x_start, y_start = 0, img_w + img_h - start - 2
                
            # Get diagonal values
            x_indices = []
            y_indices = []
            x, y = x_start, y_start
            while x < img_w and y >= 0:
                x_indices.append(x)
                y_indices.append(y)
                x += 1
                y -= 1
                
            if not x_indices:  # Skip if diagonal is empty
                continue
                
            # Convert to numpy arrays for vectorized operations
            x_indices = np.array(x_indices)
            y_indices = np.array(y_indices)
            diag_values = k4[y_indices, x_indices]
            
            # Find positions where k4 > 0
            positive_mask = diag_values > 0
            
            # Find the boundaries of continuous positive regions
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # Process each continuous region
            for s, e in zip(starts, ends):
                # Get region coordinates
                region_x = x_indices[s:e]
                region_y = y_indices[s:e]
                region = diag_values[s:e]
                Wr = len(region)
                
                if Wr > 0:
                    # Get max value and its position
                    max_val = np.max(region)
                    pos_max = np.argmax(region)
                    
                    # Calculate actual position in original array
                    max_x = region_x[pos_max]
                    max_y = region_y[pos_max]
                    
                    # Update Vt
                    Vt[max_y, max_x] += max_val * Wr

        return Vt
