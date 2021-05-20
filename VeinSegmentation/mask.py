import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from VeinSegmentation import enhance
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


def apply_skeletonization_to_roi(image, mask):
    """
    Extracción de la región de interés a partir de una máscara
    y aplicación del algoritmo de skeletonización
    """

    print("Processing apply_skeletonization_to_roi...")
    now = time()

    mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    # image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)
    skeletonized_roi = enhance.skeletonization(result)

    plt.imshow(mask, cmap="gray")
    plt.show()
    plt.imshow(skeletonized_roi, cmap="gray")
    plt.show()
    cv2.imwrite('./mask.png', mask)

    height, width = mask.shape[:2]
    enhanced = mask.copy()
    for x in range(0, height):
        for y in range(0, width):
            if mask[x, y] == white:
                enhanced[x, y] = skeletonized_roi[x, y]
            else:
                enhanced[x, y] = image[x, y]

    elapsed = time() - now
    print("Processing time: ", elapsed)
    return enhanced


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
    #image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    result = cv2.bitwise_and(image, image, mask=mask)

    edges = subpixel_edges(result, threshold, iters, order)
    points = zip(edges.x, edges.y) # TODO esto no funciona

    for point in points:
        cv2.circle(image, tuple(point), 1, (0, 0, 255)) # TODO Aqui peta porque necesitamos un array de tuplas y no el zip

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
