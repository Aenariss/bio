"""
BIO Project 2024

This module contains a class to extract features using optimized max curvature method 
Author: Filip Brna <xbrnaf00> + ChatGPT
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

        # Compute Gaussian kernel and its derivatives
        # Gaussian kernel - used for smoothing the image to prepare it for derivative computation
        # Derivatives are used to find the curvature
        h = (1 / (2 * np.pi * sigma2)) * np.exp(-X2Y2 / (2 * sigma2)) # Gaussian Function https://en.wikipedia.org/wiki/Gaussian_function
        # First derivative of the Gaussian in the x direction
        hx = -(X / sigma2) * h
        # Second derivative of the Gaussian in the x direction
        hxx = ((X2 - sigma2) / sigma4) * h
        # Transpose the derivatives for y direction
        hy = hx.T  # First derivative of Gaussian in y direction
        hyy = hxx.T  # Second derivative of Gaussian in y direction
        # Mixed second derivative of the Gaussian (xy direction)
        hxy = (X * Y / sigma4) * h

        # Apply Gaussian derivatives to the source image using 2D convolution
        # fx, fy are the first derivatives of the image in x and y directions

        # STEP 1. 1,2,3,4
        fx = -cv2.filter2D(src, -1, hx)
        fy = cv2.filter2D(src, -1, hy)
        # fxx, fyy are the second derivatives of the image in x and y directions
        fxx = cv2.filter2D(src, -1, hxx)
        fyy = cv2.filter2D(src, -1, hyy)
        # fxy is the mixed derivative (xy direction)
        fxy = -cv2.filter2D(src, -1, hxy)
        
        # Compute diagonal derivatives using fx, fy
        f1 = 0.5 * np.sqrt(2.0) * (fx + fy)
        f2 = 0.5 * np.sqrt(2.0) * (fx - fy)
        # Compute diagonal second derivatives
        f11 = 0.5 * fxx + fxy + 0.5 * fyy
        f22 = 0.5 * fxx - fxy + 0.5 * fyy

        # Initialize four curvature matrices, k1, k2, k3, k4 with zeros
        k1 = np.zeros_like(src)
        k2 = np.zeros_like(src)
        k3 = np.zeros_like(src)
        k4 = np.zeros_like(src)

         # Compute curvatures at each pixel
        for y in range(src.shape[0]):
            for x in range(src.shape[1]):
                if mask[y, x] > 0:
                    k1[y, x] = fxx[y, x] / np.power(1 + fx[y, x]**2, 1.5)
                    k2[y, x] = fyy[y, x] / np.power(1 + fy[y, x]**2, 1.5)
                    k3[y, x] = f11[y, x] / np.power(1 + f1[y, x]**2, 1.5)
                    k4[y, x] = f22[y, x] / np.power(1 + f2[y, x]**2, 1.5)

        # Scores initialization (Wr and Vt)
        img_h, img_w = src.shape[:2]
        Vt = np.zeros_like(src)

        # Horizontal direction processing
        for y in range(img_h):
            Wr = 0
            for x in range(img_w):
                if k1[y, x] > 0:
                    Wr += 1
                if Wr > 0 and (x == img_w - 1 or k1[y, x] <= 0):
                    pos_end = x if x == img_w - 1 else x - 1
                    pos_start = pos_end - Wr + 1
                    max_k1 = np.max(k1[y, pos_start:pos_end + 1])
                    pos_max = pos_start + np.argmax(k1[y, pos_start:pos_end + 1])
                    Vt[y, pos_max] += max_k1 * Wr
                    Wr = 0

        # Vertical direction processing
        for x in range(img_w):
            Wr = 0
            for y in range(img_h):
                if k2[y, x] > 0:
                    Wr += 1
                if Wr > 0 and (y == img_h - 1 or k2[y, x] <= 0):
                    pos_end = y if y == img_h - 1 else y - 1
                    pos_start = pos_end - Wr + 1
                    max_k2 = np.max(k2[pos_start:pos_end + 1, x])
                    pos_max = pos_start + np.argmax(k2[pos_start:pos_end + 1, x])
                    Vt[pos_max, x] += max_k2 * Wr
                    Wr = 0

        # Diagonal (\) direction processing
        for start in range(img_h + img_w - 1):
            x, y = (start, 0) if start < img_w else (0, start - img_w + 1)
            Wr = 0
            while x < img_w and y < img_h:
                if k3[y, x] > 0:
                    Wr += 1
                if Wr > 0 and (x == img_w - 1 or y == img_h - 1 or k3[y, x] <= 0):
                    pos_x_end, pos_y_end = (x, y) if x == img_w - 1 or y == img_h - 1 else (x - 1, y - 1)
                    pos_x_start, pos_y_start = pos_x_end - Wr + 1, pos_y_end - Wr + 1
                    max_k3 = np.max(np.diag(k3[pos_y_start:pos_y_end + 1, pos_x_start:pos_x_end + 1]))
                    pos_max = np.argmax(np.diag(k3[pos_y_start:pos_y_end + 1, pos_x_start:pos_x_end + 1]))
                    Vt[pos_y_start + pos_max, pos_x_start + pos_max] += max_k3 * Wr
                    Wr = 0
                x += 1
                y += 1

        # Diagonal (/) direction processing
        for start in range(img_h + img_w - 1):
            x, y = (start, img_h - 1) if start < img_w else (0, img_w + img_h - start - 2)
            Wr = 0
            while x < img_w and y >= 0:
                if k4[y, x] > 0:
                    Wr += 1
                if Wr > 0 and (x == img_w - 1 or y == 0 or k4[y, x] <= 0):
                    pos_x_end, pos_y_end = (x, y) if x == img_w - 1 or y == 0 else (x - 1, y + 1)
                    pos_x_start, pos_y_start = pos_x_end - Wr + 1, pos_y_end + Wr - 1
                    max_k4 = np.max(np.diag(np.flipud(k4[pos_y_end:pos_y_start + 1, pos_x_start:pos_x_end + 1])))
                    pos_max = np.argmax(np.diag(np.flipud(k4[pos_y_end:pos_y_start + 1, pos_x_start:pos_x_end + 1])))
                    Vt[pos_y_start - pos_max, pos_x_start + pos_max] += max_k4 * Wr
                    Wr = 0
                x += 1
                y -= 1

        return Vt
