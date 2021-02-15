import cv2
from anisotropic_difussion import anisodiff
import numpy as np


# Based in https://ieeexplore.ieee.org/document/6246971

# Input: Original image
# Steps:
#   0: Denoise
#   1: Mathematical Morphology
#   2: Anisotropic Diffusion Filter
#   3: Contrast Limited Histogram Equalization
# Output: Enhanced Medical Image

def enhance_medical_image(image, clip_limit=10, tile_grid_size=20):
    image = cv2.fastNlMeansDenoising(image.astype(np.uint8))
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    image = anisodiff(image)
    image = clahe.apply(image.astype(np.uint8))
    del clahe
    return image.astype(float)
