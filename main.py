import copy
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import time

from VeinSegmentation.contour import contour
from VeinSegmentation.enhance import anisodiff, enhance_medical_image, segmentation, skeletonization, gaborFiltering
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

white = 255.
black = 0.

if __name__ == '__main__':
    this_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(this_path, 'imagenes_orginales/Caso A BN.png'))

    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
    zoom = img_gray[y:y + h, x:x + w]

    enhanced = enhance_medical_image(zoom, clip_limit=8, tile_grid_size=8)
    enhanced_segm = segmentation(enhanced, n_clusters=3)
    plt.title("Segmented")
    plt.imshow(enhanced_segm, cmap='gray')
    plt.show()

    colors = np.unique(enhanced_segm)  # De mas oscuro a mas claro
    print(colors)

    # TODO PASARLO A UN METODO

    each_color_picture = []  # Con el valor a 1 y el resto a 0

    for value in colors:
        color_layer = copy.deepcopy(enhanced_segm)  # Con el valor a 1 y el resto a 0
        h = enhanced_segm.shape[0]
        w = enhanced_segm.shape[1]

        # iteramos sobre cada pixel
        for y in range(0, h):
            for x in range(0, w):
                if enhanced_segm[y, x] == value:
                    color_layer[y, x] = white
                else:
                    color_layer[y, x] = black

        each_color_picture.append(color_layer)

    # TODO PASARLO A UN METODO
    # TODO SACAR LOS ESQUELETOS
    each_filled_picture = []  # Con el valor y los menores al valor a 1 y el resto a 0

    for value in colors:
        filled_color_layer = copy.deepcopy(enhanced_segm)

        h = enhanced_segm.shape[0]
        w = enhanced_segm.shape[1]

        # iteramos sobre cada pixel
        for y in range(0, h):
            for x in range(0, w):
                if enhanced_segm[y, x] <= value:
                    filled_color_layer[y, x] = white
                else:
                    filled_color_layer[y, x] = black

        if cv2.countNonZero(filled_color_layer) != 0:  # Desechamos las imagenes que sean completamente negras
            each_filled_picture.append(filled_color_layer)
            plt.imshow(filled_color_layer, cmap='gray')
            plt.title("Filled Layer {}".format(value))
            plt.show()

    ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
    inverted_segm_th = cv2.bitwise_not(img)
    skel = skeletonization(inverted_segm_th)

    open_contours, closed_contours = contour(enhanced_segm.astype(np.uint8))
    black = np.zeros((h, w, 3), np.uint8)

    black_countour = cv2.drawContours(black, open_contours, -1, (255, 255, 255), 3)

    gabor_filtered_enhanced = gaborFiltering(enhanced)

    # print("Processing original image...")
    # now = time()
    # edges = subpixel_edges(zoom, threshold, iters, order)
    # elapsed = time() - now
    # print("Processing time: ", elapsed)
