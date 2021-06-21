import cv2
import numpy as np
from VeinSegmentation.Enhance import segmentation
import matplotlib.pyplot as plt


def skeletonization(img, niter=100):
    enhanced_segm = segmentation(img, n_clusters=2)

    ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
    img = cv2.bitwise_not(img)

    img = img.astype(np.uint8)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Kernel con forma de cruz
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    i = 0
    while i < niter:
        i = i + 1
        # Substraemos la apertura de la imagen
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        # Erosionamos y refinamos el esqueleto
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Si no quedan píxeles blancos, la imagen ya ha sido esqueletonizada
        if cv2.countNonZero(img) == 0:
            break
    return skel
