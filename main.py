import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

from VeinSegmentation.contour import contour
from VeinSegmentation.enhance import anisodiff, enhance_medical_image, segmentation, skeletonization
from subpixel_edges import init, subpixel_edges

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 1.5
order = 2

if __name__ == '__main__':
    init()
    this_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(this_path, 'imagenes_orginales/Caso A BN.png'))

    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
    zoom = img_gray[y:y + h, x:x + w]

    enhanced = enhance_medical_image(zoom, clip_limit=8, tile_grid_size=8)
    enhanced_segm = segmentation(enhanced, n_clusters=2)

    ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
    inverted_segm_th = cv2.bitwise_not(img)
    skel = skeletonization(inverted_segm_th)

    open_contours, closed_contours = contour(enhanced_segm.astype(np.uint8))
    black = np.zeros((h, w, 3), np.uint8)

    black_countour = cv2.drawContours(black, open_contours, -1, (255, 255, 255), 3)
    contour_image = cv2.drawContours(zoom, open_contours, -1, (255, 0, 0), 3)

    # print("Processing original image...")
    # now = time()
    # edges = subpixel_edges(zoom, threshold, iters, order)
    # elapsed = time() - now
    # print("Processing time: ", elapsed)

    plt.imshow(zoom, cmap="gray")
    plt.title("Orginal")
    plt.show()
    plt.title("Mejorada")
    plt.imshow(enhanced, cmap='gray')
    plt.show()
    plt.title("Mejorada y segmentada")
    plt.imshow(enhanced_segm, cmap='gray')
    plt.show()
    plt.title("Contorno sobre original")
    plt.imshow(contour_image, cmap='gray')
    plt.show()
    plt.title("Contorno sobre negro")
    plt.imshow(black_countour, cmap='gray')
    plt.show()
    plt.title('Esqueleto')
    plt.imshow(skel, cmap='gray')
    plt.show()

