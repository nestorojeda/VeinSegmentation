from time import time

import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from VeinSegmentation import Enhance
from VeinSegmentation.Skeletonization import skeletonization


def getMaskArea(mask):
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    area = 0
    for cnt in contours:
        area += cv2.contourArea(cnt)

    return area


def applyEnhanceToROI(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicacion de la mejora para imagenes médicas
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_enhance_to_roi...")
    now = time()

    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]  # Corte que contiene el poligono maximo
        enhancedCrop = Enhance.enhanceMedicalImage(crop)  # Corte mejorado
        blackPixels = cv2.countNonZero(image)

        merged = image.copy()
        enhancedCrop = enhancedCrop.astype(np.uint8)
        merged[y:y + h, x:x + w] = enhancedCrop  # original con el corte superpuesto
        result = processResult(image, merged, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2RGB), blackPixels


def applySkeletonizationToROI(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicación del algoritmo de skeletonización
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_skeletonization_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        crop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)

        skel_crop = skeletonization(crop)
        whitePixels = np.sum(skel_crop == 255)

        merged = image.copy()
        skel_crop = cv2.cvtColor(skel_crop.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        merged[y:y + h, x:x + w] = skel_crop
        result = processResult(image, merged, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return result, whitePixels


def applySubpixelToROI(image, mask,
                       iters=2,
                       threshold=1.5,
                       order=2
                       ):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicacion del algoritmo de subpixel
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_subpixel_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        crop = Enhance.enhanceMedicalImage(crop)
        edges = subpixel_edges(crop.astype(float), threshold, iters, order)

        edgedCrop = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for point in np.array((edges.x, edges.y)).T.astype(np.uint):
            cv2.circle(edgedCrop, tuple(point), 1, (0, 0, 255))

        merged = image.copy()
        merged[y:y + h, x:x + w] = edgedCrop
        result = processResult(image, merged, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return result.astype(np.uint8)


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
        result = processResult(image, merged, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return result


def processResult(image, merged, mask):
    fg = cv2.bitwise_or(merged, merged, mask=mask)
    mask = cv2.bitwise_not(mask)
    fgbg = cv2.bitwise_or(fg, image, mask=mask)
    result = cv2.bitwise_or(fgbg, fg)
    return result
