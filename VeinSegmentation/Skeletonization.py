import cv2
import numpy as np

from VeinSegmentation.Enhance import quantification


def cleanSkeleton(skeleton):
    h = skeleton.shape[0]
    w = skeleton.shape[1]

    new = np.zeros((h, w))
    for i in range(0, h):
        for j in range(0, w):
            region = skeleton[j:j + 3, i:i + 3]
            if sum(sum(region == 255)) == 3:
                new[j:j + 3, i:i + 3] = region

    return new


def skeletonization(img, niter=100):
    segm = quantification(img, n_clusters=2)

    ret, img = cv2.threshold(segm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
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
