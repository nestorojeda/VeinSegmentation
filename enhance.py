import cv2
from anisotropic_difussion import anisodiff
import numpy as np


# Based in https://ieeexplore.ieee.org/document/6246971

# Input: Original image
# Steps:
#   1: Mathematical Morphology
#   2: Anisotropic Diffusion Filter
#   3: Contrast Limited Histogram Equalization
# Output: Enhanced Medical Image

def enhance_medical_image(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    image = anisodiff(image)
    image = clahe.apply(image.astype(np.uint8))
    del clahe
    return image.astype(float)

