import cv2
import numpy as np


def cleanSkeleton(skeleton):
    h = skeleton.shape[0]
    w = skeleton.shape[1]

    # Limpieza de los pixeles espureos
    new = np.zeros((h, w))
    for j in range(0, h):
        for i in range(0, w):
            region = skeleton[j:j + 3, i:i + 3]
            if sum(sum(region == 255)) == 3:
                new[j:j + 3, i:i + 3] = region

    # Esto limpia los pixeles en L de las líneas diagonales
    for i in range(0, h):
        for j in range(0, w):
            if new[i, j] == 255:
                if (i + 1) < h and (j + 1) < w:
                    if new[i + 1, j + 1] == 255:
                        if new[i + 1, j] == 255:
                            new[i + 1, j] = 0
                        if new[i, j + 1] == 255:
                            new[i, j + 1] = 0

                if (i + 1) < h and (j - 1) >= 0:
                    if new[i + 1, j - 1] == 255:
                        if new[i + 1, j] == 255:
                            new[i + 1, j] = 0
                        if new[i, j - 1] == 255:
                            new[i, j - 1] = 0

    return new


def skeletonization(img, niter=100):
    ret, img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(img)

    img = img.astype(np.uint8)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    return skel, contours


def skeletonLenght(skeleton, pixelSize):
    h = skeleton.shape[0]
    w = skeleton.shape[1]
    measure = 0
    for i in range(0, h):
        for j in range(0, w):
            if skeleton[i, j] == 255:
                # Diagonal
                if (i + 1) < h and (j + 1) < w:
                    if skeleton[i + 1, j + 1] == 255:
                        measure += np.sqrt(2) * pixelSize
                # Vertical
                if (i + 1) < h:
                    if skeleton[i + 1, j] == 255:
                        measure += pixelSize
                # Horizontal
                if (j + 1) < w:
                    if skeleton[i, j + 1] == 255:
                        measure += pixelSize

    return measure