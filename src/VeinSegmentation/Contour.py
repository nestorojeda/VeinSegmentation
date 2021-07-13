import cv2
import imutils
import numpy as np


def contour(image):

    if len(image.shape) == 3:
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)).astype(np.uint8)
    else:
        gray = image.astype(np.uint8)

    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    contours = cv2.findContours(edged.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # loop over our contours
    closedContours = []
    openContours = []

    for c in contours:
        if cv2.contourArea(c) > cv2.arcLength(c, True):
            closedContours.append(c)
        else:
            openContours.append(c)

    return openContours, closedContours


def sortContours(contours):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    closedContours = []
    openContours = []

    for c in contours:
        if cv2.contourArea(c) > cv2.arcLength(c, True):
            closedContours.append(c)
        else:
            openContours.append(c)

    return openContours, closedContours
