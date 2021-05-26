import cv2
import matplotlib.pyplot as plt
import numpy as np

from VeinSegmentation import enhance as eh
from subpixel_edges import subpixel_edges

iters = 2
threshold = 1.5
order = 2

image = cv2.imread('imagenes_orginales/Caso A BN.png')
mask = cv2.imread('./mask.png', cv2.IMREAD_GRAYSCALE)

contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
idx = 0
for cnt in contours:
    idx += 1
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]
    enhanced_crop = eh.enhance_medical_image(crop)

    subpixel_edges_crop = subpixel_edges(enhanced_crop, threshold, iters, order)

    plt.imshow(subpixel_edges_crop, cmap='gray')
    plt.title('subpixel_edges crop')
    plt.show()

    merged = image.copy()
    subpixel_edges_crop = cv2.cvtColor(subpixel_edges_crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    merged[y:y + h, x:x + w] = subpixel_edges_crop

    plt.imshow(merged, cmap='gray')
    plt.title('merged')
    plt.show()

    # get first masked value (foreground)
    fg = cv2.bitwise_or(merged, merged, mask=mask)
    plt.imshow(fg, cmap='gray')
    plt.title('fg')
    plt.show()
    # get second masked value (background) mask must be inverted
    mask = cv2.bitwise_not(mask)

    # combine foreground+background
    test = cv2.bitwise_or(fg, image, mask=mask)

    plt.imshow(test, cmap='gray')
    plt.title('test')
    plt.show()

    mask = cv2.bitwise_not(mask)
    final = cv2.bitwise_or(test, fg)

    plt.imshow(final, cmap='gray')
    plt.title('final')
    plt.show()

    # TODO PASAR ESTO A UN MODULO
