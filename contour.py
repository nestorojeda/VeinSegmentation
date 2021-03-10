import cv2
import imutils


def contour(image):
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

    ratio = image.shape[0] / 300.0
    orig = image.copy()
    image = imutils.resize(image, height=300)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    contour_image = cv2.drawContours(image, open_contours, -1, (255, 0, 0), 3)
    return contour_image;

