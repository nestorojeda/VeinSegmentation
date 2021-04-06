import copy
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from VeinSegmentation.enhance import enhance_medical_image, segmentation, smooth_thresholded_image, skeletonization, \
    color_layer_segmentation, color_layer_segmantation_filled
from plotting import plotArray

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
    plt.imshow(zoom, cmap='gray')
    plt.title("Original")
    plt.show()

    enhanced = enhance_medical_image(zoom, clip_limit=8, tile_grid_size=8)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced")
    plt.show()

    enhanced_segm = segmentation(enhanced, n_clusters=10)
    plt.title("Segmented")
    plt.imshow(enhanced_segm, cmap='gray')
    plt.show()

    each_filled_picture = color_layer_segmantation_filled(enhanced_segm)

    # SMOOTHING
    smoothed_images = []
    for image in each_filled_picture:
        smoothed = smooth_thresholded_image(image)
        # plt.imshow(smoothed, cmap='gray')
        # plt.title("Smoothed skeleton")
        # plt.show()
        smoothed_images.append(smoothed)

    # SKELETON
    skeletons = []
    for image in smoothed_images:
        skel = skeletonization(image)
        skeletons.append(skel)
        # plt.imshow(skel, cmap='gray')
        # plt.title("Skeleton")
        # plt.show()
        del skel

    # ENHANCE SKELETONS
    enhanced_skeletons = []
    for image in skeletons:
        enhanced_skeleton = np.zeros((zoom.shape[0], zoom.shape[1]))
        h = zoom.shape[0]
        w = zoom.shape[1]
        for y in range(0, h):
            for x in range(0, w):
                if image[y, x] != black:
                    enhanced_skeleton[y, x] = white

        enhanced_skeletons.append(enhanced_skeleton)
        # plt.imshow(enhanced_skeleton, cmap='gray')
        # plt.title("Enhanced skeleton")
        # plt.show()
        del enhanced_skeleton

    # MERGE SKELETON
    canvas = np.zeros((zoom.shape[0], zoom.shape[1]))
    for image in enhanced_skeletons:
        h = zoom.shape[0]
        w = zoom.shape[1]
        for y in range(0, h):
            for x in range(0, w):
                if image[y, x] == white:
                    canvas[y, x] = white

    plt.imshow(canvas, cmap='gray')
    plt.title("Merged skeleton")
    plt.show()

    plotArray(each_filled_picture)

    # print("Processing original image...")
    # now = time()
    # edges = subpixel_edges(zoom, threshold, iters, order)
    # elapsed = time() - now
    # print("Processing time: ", elapsed)
