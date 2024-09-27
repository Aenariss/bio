"""
BIO Project 2024

This module contains a class to extract features using optimized max curvature method 
Author: Filip Brna <xbrnaf00> + ChatGPT
"""

import numpy as np
import cv2

class MaxCurvature:
    def __init__(self):
        pass

    def meshgrid(self, xgv, ygv):
        X, Y = np.meshgrid(xgv, ygv)
        return X, Y

    def max_curvature(self, src, mask, sigma=8):
        src = src.astype(np.float32) / 255.0
        sigma2 = sigma ** 2
        sigma4 = sigma ** 4

        winsize = int(np.ceil(4 * sigma))
        X, Y = self.meshgrid(np.arange(-winsize, winsize + 1), np.arange(-winsize, winsize + 1))

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
