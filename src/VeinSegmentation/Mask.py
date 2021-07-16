from time import time

import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from src.VeinSegmentation import Enhance


def applyBrightnessAndContrastToROI(image, mask, brightness, contrast):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicacion de la mejora para imagenes médicas
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_enhance_to_roi...")
    now = time()
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]  # Corte que contiene el poligono maximo
        enhanced_crop = Enhance.processBrightnessAndContrast(crop, brightness, contrast)  # Corte mejorado

        merged = image.copy()
        enhanced_crop = enhanced_crop.astype(np.uint8)
        merged[y:y + h, x:x + w] = enhanced_crop  # original con el corte superpuesto
        result = processResult(image, crop, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return result


def processResult(image, merged, mask):
    """ Creacion del resultado mediante operaciones lógicas """
    fg = cv2.bitwise_or(merged, merged, mask=mask)
    mask = cv2.bitwise_not(mask)
    fgbg = cv2.bitwise_or(fg, image, mask=mask)
    result = cv2.bitwise_or(fgbg, fg)
    return result
