import cv2
from VeinSegmentation.Contour import contour
import matplotlib.pyplot as plt
import numpy as np
from VeinSegmentation.Enhance import enhanceMedicalImage, quantification, skeletonization

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

image = (cv2.imread('../imagenes_orginales/Caso A BN.png'))
zoom = image[y:y + h, x:x + w]

enhanced = enhanceMedicalImage(zoom, clip_limit=8, tile_grid_size=8)
enhanced_segm = quantification(enhanced, n_clusters=2)

ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
inverted_segm_th = cv2.bitwise_not(img)
skel = skeletonization(inverted_segm_th)

open_contours, closed_contours = contour(enhanced_segm.astype(np.uint8))
black = np.zeros((h, w, 3), np.uint8)

black_countour = cv2.drawContours(black, open_contours, -1, (255, 255, 255), 3)
contour_image = cv2.drawContours(zoom, open_contours, -1, (255, 0, 0), 3)

plt.title("Mejorada")
plt.imshow(enhanced, cmap='gray')
plt.show()
plt.title("Mejorada y segmentada")
plt.imshow(enhanced_segm, cmap='gray')
plt.show()
plt.title("Contorno sobre original")
plt.imshow(contour_image, cmap='gray')
plt.show()
plt.title("Contorno sobre negro")
plt.imshow(black_countour, cmap='gray')
plt.show()
plt.title('Esqueleto')
plt.imshow(skel, cmap='gray')
plt.show()