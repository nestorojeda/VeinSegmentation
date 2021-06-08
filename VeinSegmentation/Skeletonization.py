import cv2
import numpy as np
from VeinSegmentation.Enhance import segmentation

def skeletonization(img, niter=100):
    enhanced_segm = segmentation(img, n_clusters=2)

    ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
    img = cv2.bitwise_not(img)

    img = img.astype(np.uint8)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    i = 0
    while i < niter:
        i = i + 1
        # Step 1: Substract open from the original image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        # Step 2: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 3: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break
    return skel
