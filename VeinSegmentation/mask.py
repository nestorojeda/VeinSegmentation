import cv2
import numpy as np


def extract_roi(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    https://www.programmersought.com/article/75844449435/
    """
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

