from time import time

import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from src.VeinSegmentation import Enhance, Contour
from src.VeinSegmentation.Skeletonization import skeletonization, cleanSkeleton


def getMaskArea(mask):
    """ Obtención del numero de pixeles de la imagen"""
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
        enhancedCrop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)

        skelCrop, contours = skeletonization(enhancedCrop)
        # Mejoramos el esqueleto solo para hallar el verdadero trazado de la vena
        cleanedSkeleton = cleanSkeleton(skelCrop)

        skelCrop = cv2.cvtColor(skelCrop.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        skelCropTransparent = skelCrop.copy()
        cleanedSkeleton = cv2.cvtColor(cleanedSkeleton.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        for j in range(0, skelCrop.shape[0]):
            for i in range(0, skelCrop.shape[1]):
                if np.array_equal(skelCrop[j, i], np.array([255, 255, 255])):
                    skelCrop[j, i] = (0, 0, 255)
                    skelCropTransparent[j, i] = (0, 0, 255)
                if np.array_equal(skelCrop[j, i], np.array([0, 0, 0])):
                    skelCropTransparent[j, i] = crop[j, i]

        openContours, closedContours = Contour.sortContours(contours)
        skelCropWithContours = cv2.drawContours(skelCrop.copy(), closedContours, -1, (0, 255, 0), 1)
        skelCropWithContoursAndTransparency = cv2.drawContours(skelCropTransparent.copy(), closedContours, -1,
                                                               (0, 255, 0), 1)

        merged = image.copy()
        merged[y:y + h, x:x + w] = skelCrop

        mergedWithTransparency = image.copy()
        mergedWithTransparency[y:y + h, x:x + w] = skelCropTransparent

        mergedWithContours = image.copy()
        mergedWithContours[y:y + h, x:x + w] = skelCropWithContours

        mergedWithContoursAndTransparency = image.copy()
        mergedWithContoursAndTransparency[y:y + h, x:x + w] = skelCropWithContoursAndTransparency

        result = processResult(image, merged, mask)
        resultWithContours = processResult(image, mergedWithTransparency, mask)
        resultWithTransparency = processResult(image, mergedWithContours, mask)
        resultWithContoursAndTransparency = processResult(image, mergedWithContoursAndTransparency, mask)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return result, resultWithContours, resultWithTransparency, resultWithContoursAndTransparency, cleanedSkeleton


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
