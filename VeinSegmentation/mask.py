import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from VeinSegmentation import enhance as eh
import matplotlib.pyplot as plt
from time import time

white = 255.
black = 0.


def apply_enhance_to_roi(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicacion de la mejora para imagenes médicas
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_enhance_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]  # Corte que contiene el poligono maximo
        enhanced_crop = eh.enhance_medical_image(crop)  # Corte mejorado

        merged = image.copy()
        enhanced_crop = enhanced_crop.astype(np.uint8)
        merged[y:y + h, x:x + w] = enhanced_crop  # original con el corte superpuesto

        fg = cv2.bitwise_or(merged, merged, mask=mask)  # la parte que ha sido mejorada

        mask = cv2.bitwise_not(mask)  # cambiamos la mascara de signo

        hollow_picture = cv2.bitwise_or(fg, image, mask=mask)  # la imagen con un agujero

        mask = cv2.bitwise_not(mask)
        enhanced = cv2.bitwise_or(hollow_picture, fg)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return enhanced


def apply_skeletonization_to_roi(image, mask, is_enhanced=True):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicación del algoritmo de skeletonización
    """

    print("Processing apply_skeletonization_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    if not is_enhanced: image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    idx = 0
    for cnt in contours:
        idx += 1
        x, y, w, h = cv2.boundingRect(cnt)
        crop = image[y:y + h, x:x + w]
        if not is_enhanced: crop = eh.enhance_medical_image(
            crop)  # Si la imagen no está mejorada, la mejoramos solo para realizar este proceso

        skel_crop = eh.skeletonization(crop)

        plt.imshow(skel_crop, cmap='gray')
        plt.title('skel crop')
        plt.show()

        merged = image.copy()
        skel_crop = skel_crop.astype(np.uint8)
        merged[y:y + h, x:x + w] = skel_crop

        # get first masked value (foreground)
        fg = cv2.bitwise_or(merged, merged, mask=mask)
        # get second masked value (background) mask must be inverted
        mask = cv2.bitwise_not(mask)

        # combine foreground+background
        test = cv2.bitwise_or(fg, image, mask=mask)

        mask = cv2.bitwise_not(mask)
        skeleton = cv2.bitwise_or(test, fg)

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return skeleton


def apply_subpixel_to_roi(image, mask,
                          iters=2,
                          threshold=1.5,
                          order=2):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicacion del algoritmo de subpixel
    https://www.programmersought.com/article/75844449435/
    """

    print("Processing apply_subpixel_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)

    edges = subpixel_edges(result, threshold, iters, order)
    points = zip(edges.x, edges.y)  # TODO esto no funciona

    for point in points:
        cv2.circle(image, tuple(point), 1,
                   (0, 0, 255))  # TODO Aqui peta porque necesitamos un array de tuplas y no el zip

    height, width = mask.shape[:2]
    enhanced = mask.copy()
    for x in range(0, height):
        for y in range(0, width):
            if mask[x, y] == white:
                enhanced[x, y] = image[x, y]
            else:
                enhanced[x, y] = image[x, y]

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return enhanced
