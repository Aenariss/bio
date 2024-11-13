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

    # Method to create a 2D meshgrid from x and y grid values
    def meshgrid(self, xgv, ygv):
        # Create two 2D arrays (grids) from x and y grid vectors using np.meshgrid
        X, Y = np.meshgrid(xgv, ygv)
        # Return the generated grid
        return X, Y

    # Method to compute the maximum curvature of the image
    def max_curvature(self, src, mask, sigma=8):

        # Normalize the source image to the range [0, 1] by dividing by 255.0
        src = src.astype(np.float32) / 255.0

        # Precompute sigma^2 and sigma^4 for use in Gaussian derivatives
        sigma2 = sigma ** 2
        sigma4 = sigma ** 4

        # Determine the window size based on the sigma value (to limit the kernel size)
        winsize = int(np.ceil(4 * sigma))

        # Generate a meshgrid of coordinates in the range of [-winsize, winsize] for both X and Y
        X, Y = self.meshgrid(np.arange(-winsize, winsize + 1), np.arange(-winsize, winsize + 1))

        # Compute X^2, Y^2, and their sum X^2 + Y^2
        X2 = np.power(X, 2)
        Y2 = np.power(Y, 2)
        X2Y2 = X2 + Y2

        # STEP 1-1: Compute Gaussian kernel and its derivatives
        h = (1 / (2 * np.pi * sigma2)) * np.exp(-X2Y2 / (2 * sigma2))  # Gaussian Function
        hx = -(X / sigma2) * h  # First derivative of the Gaussian in the x direction
        hxx = ((X2 - sigma2) / sigma4) * h  # Second derivative of the Gaussian in the x direction
        hy = hx.T  # First derivative of Gaussian in y direction
        hyy = hxx.T  # Second derivative of Gaussian in y direction
        hxy = (X * Y / sigma4) * h  # Mixed second derivative of the Gaussian (xy direction)


        # STEP 1-2: Apply Gaussian derivatives to the source image using 2D convolution
        # fx, fy are the first derivatives of the image in x and y directions
        fx = -cv2.filter2D(src, -1, hx)  # First derivative in the x direction
        fy = cv2.filter2D(src, -1, hy)  # First derivative in the y direction
        fxx = cv2.filter2D(src, -1, hxx)  # Second derivative in the x direction
        fyy = cv2.filter2D(src, -1, hyy)  # Second derivative in the y direction
        fxy = -cv2.filter2D(src, -1, hxy)  # Mixed second derivative (xy direction)
        
        # STEP 1-3: Compute diagonal derivatives using fx, fy
        f1 = 0.5 * np.sqrt(2.0) * (fx + fy)  # First diagonal derivative
        f2 = 0.5 * np.sqrt(2.0) * (fx - fy)  # Second diagonal derivative

        # Compute diagonal second derivatives
        f11 = 0.5 * fxx + fxy + 0.5 * fyy  # Second derivative along one diagonal
        f22 = 0.5 * fxx - fxy + 0.5 * fyy  # Second derivative along the other diagonal

        # STEP 1-4: Initialize four curvature matrices, k1, k2, k3, k4 with zeros
        k1 = np.zeros_like(src)
        k2 = np.zeros_like(src)
        k3 = np.zeros_like(src)
        k4 = np.zeros_like(src)

        # Create mask for valid pixels
        valid_mask = mask > 0

        # Precalculate denominators
        denom1 = np.power(1 + fx**2, 1.5)
        denom2 = np.power(1 + fy**2, 1.5)
        denom3 = np.power(1 + f1**2, 1.5)
        denom4 = np.power(1 + f2**2, 1.5)

        # Compute curvatures at each pixel of the ROI
        k1[valid_mask] = fxx[valid_mask] / denom1[valid_mask]
        k2[valid_mask] = fyy[valid_mask] / denom2[valid_mask]
        k3[valid_mask] = f11[valid_mask] / denom3[valid_mask]
        k4[valid_mask] = f22[valid_mask] / denom4[valid_mask]


        '''
        # Compute curvatures at each pixel
        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                if mask[y, x] > 0:  # Only compute curvature for the masked area
                    k1[y, x] = fxx[y, x] / np.power(1 + fx[y, x] ** 2, 1.5)  # Curvature in x direction
                    k2[y, x] = fyy[y, x] / np.power(1 + fy[y, x] ** 2, 1.5)  # Curvature in y direction
                    k3[y, x] = f11[y, x] / np.power(1 + f1[y, x] ** 2, 1.5)  # Curvature along one diagonal
                    k4[y, x] = f22[y, x] / np.power(1 + f2[y, x] ** 2, 1.5)  # Curvature along the other diagonal
        '''

        # Step 2 - vein connections
        # Scores initialization (Wr and Vt)
        img_h, img_w = src.shape[:2]
        Vt = np.zeros_like(src)

        # Horizontal direction processing
        # Vectorized approach using boolean masks and numpy operations
        for y in range(img_h):
            # Find positions where k1 > 0
            positive_mask = k1[y, :] > 0
            
            # Find the boundaries of continuous positive regions
            # This detects changes in the mask
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # Process each continuous region
            for start, end in zip(starts, ends):
                region = k1[y, start:end]
                Wr = len(region)  # Width of the region
                
                # Find maximum value and its position in the region
                max_val = np.max(region)
                pos_max = start + np.argmax(region)
                
                # Update Vt at the maximum position
                Vt[y, pos_max] += max_val * Wr

        # Vertical direction processing
        # Vectorized approach using boolean masks and numpy operations
        for x in range(img_w):
            # Find positions where k2 > 0
            positive_mask = k2[:, x] > 0
            
            # Find the boundaries of continuous positive regions
            # This detects changes in the mask
            changes = np.diff(positive_mask.astype(int), prepend=0, append=0)
            starts = np.where(changes == 1)[0]
            ends = np.where(changes == -1)[0]
            
            # Process each continuous region
            for start, end in zip(starts, ends):
                region = k2[start:end, x]
                Wr = len(region)  # Width of the region
                
                # Find maximum value and its position in the region
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
