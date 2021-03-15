import cv2
import imutils
import matplotlib.pyplot as plt


def contour(image):
    if len(image.shape) == 3:
        gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    else:
        gray = image

    ratio = image.shape[0] / 300.0
    orig = image.copy()

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)

    # find contours in the edged image, keep only the largest
    # ones, and initialize our screen contour
    contours = cv2.findContours(edged.copy(), mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # loop over our contours
    closed_contours = []
    open_contours = []

    for c in contours:
        if cv2.contourArea(c) > cv2.arcLength(c, True):
            closed_contours.append(c)
        else:
            open_contours.append(c)

    return open_contours, closed_contours
