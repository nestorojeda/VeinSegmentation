import cv2
import numpy as np
from VeinSegmentation import enhance
import matplotlib.pyplot as plt
from time import time

white = 255.
black = 0.


def apply_enhance_to_roi(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_enhance_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)
    enhanced_roi = enhance.enhance_medical_image(result)

    height, width = mask.shape[:2]
    enhanced = mask.copy()
    for x in range(0, height):
        for y in range(0, width):
            if mask[x, y] == white:
                enhanced[x, y] = enhanced_roi[x, y]
            else:
                enhanced[x, y] = image[x, y]

    elapsed = time() - now
    print("Processing time: ", elapsed)

    return enhanced
