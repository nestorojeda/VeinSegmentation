import copy
import os
from time import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

from VeinSegmentation.enhance import enhance_medical_image, segmentation, smooth_thresholded_image, skeletonization, \
    color_layer_segmantation_filled
from utils.plotting import plotArray

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

plot_substeps = True
dissmiss_whites = False
result_over_enhanced = True
if __name__ == '__main__':
    this_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(this_path, '../imagenes_orginales/Caso A BN.png'))
    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)

    now = time()

    zoom = img_gray[y:y + h, x:x + w]
    plt.imshow(zoom, cmap='gray')
    plt.title("Original")
    plt.show()

    enhanced = enhance_medical_image(zoom, clip_limit=8, tile_grid_size=8, use_clahe=True)
    plt.imshow(enhanced, cmap='gray')
    plt.title("Enhanced")
    plt.show()

    th, im_gray_th_otsu = cv2.threshold(enhanced.astype(np.uint8), 128, 255, cv2.THRESH_OTSU)
    plt.imshow(cv2.bitwise_not(im_gray_th_otsu), cmap='gray')
    plt.title("Thresholded")
    plt.show()

    enhanced_segm = segmentation(enhanced, n_clusters=20)
    plt.title("Segmented")
    plt.imshow(enhanced_segm, cmap='gray')
    plt.show()

    each_filled_picture = color_layer_segmantation_filled(enhanced_segm)
    if plot_substeps:
        plotArray(each_filled_picture, "Filled color layers")

    filled = []
    if dissmiss_whites:
        # Desechamos las imagenes blancas
        for image in each_filled_picture:
            uniques, counts = np.unique(image, return_counts=True)
            if counts[1] < counts[0]:
                filled.append(image)
    else:
        filled = each_filled_picture

    # SMOOTHING
    smoothed_images = []
    for image in filled:
        smoothed = smooth_thresholded_image(image)
        smoothed_images.append(smoothed)

    del filled

    if plot_substeps:
        plotArray(smoothed_images, "Smoothed")

    # TO BINARY

    binarized_smoothed = []
    for image in smoothed_images:
        binary = np.zeros((zoom.shape[0], zoom.shape[1]))
        for y in range(0, h):
            for x in range(0, w):
                if image[y, x] != black:
                    binary[y, x] = white

        binarized_smoothed.append(binary)
        del binary

    if plot_substeps:
        plotArray(binarized_smoothed, "Binary smoothed")

    # SKELETON
    skeletons = []
    for image in smoothed_images:
        skel = skeletonization(image)
        skeletons.append(skel)
        del skel

    del smoothed_images

    if plot_substeps:
        plotArray(skeletons, "Skeletons")

    # ENHANCE SKELETONS
    enhanced_skeletons = []
    for image in skeletons:
        enhanced_skeleton = np.zeros((zoom.shape[0], zoom.shape[1]))
        for y in range(0, h):
            for x in range(0, w):
                if image[y, x] != black:
                    enhanced_skeleton[y, x] = white

        enhanced_skeletons.append(enhanced_skeleton)
        del enhanced_skeleton

    del skeletons

    if plot_substeps:
        plotArray(enhanced_skeletons, "Enhanced skeletons")

    # MERGE ENHANCED SKELETON
    merged_skeleton = np.zeros((zoom.shape[0], zoom.shape[1]))
    for image in enhanced_skeletons:
        for y in range(0, h):
            for x in range(0, w):
                if image[y, x] == white:
                    merged_skeleton[y, x] = white

    plt.imshow(merged_skeleton, cmap='gray')
    plt.title("Merged enhanced skeleton")
    plt.show()

    if result_over_enhanced:
        skeleton_over = copy.deepcopy(enhanced)
    else:
        skeleton_over = copy.deepcopy(zoom)
    skeleton_over = cv2.cvtColor(skeleton_over.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    for y in range(0, h):
        for x in range(0, w):
            if all(merged_skeleton[y, x] == [255, 255, 255]):
                skeleton_over[y, x] = [255, 0, 0]

    plt.imshow(skeleton_over, cmap="gray")
    plt.title("Skeleton over original")
    plt.show()
    del skeleton_over

    clean_skeleton = cv2.morphologyEx(merged_skeleton, cv2.MORPH_OPEN,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
    plt.imshow(clean_skeleton, cmap="gray")
    plt.title("Clean skeleton")
    plt.show()

    elapsed = time() - now
    print("Processing time: ", elapsed, ' s')
