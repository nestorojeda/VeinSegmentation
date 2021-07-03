import cv2
import matplotlib.pyplot as plt
import numpy as np
from VeinSegmentation.Enhance import enhanceMedicalImage, quantification
from VeinSegmentation.Skeletonization import skeletonization

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 4.5
order = 2

original = (cv2.imread('../imagenes_orginales/Caso A BN.png'))
zoom = original[y:y + h, x:x + w]
plt.imshow(zoom, cmap='gray')
plt.title('Original')
plt.show()

enhanced = enhanceMedicalImage(zoom.copy())

old = skeletonization(enhanced.copy())
plt.imshow(old, cmap='gray')
plt.title('Old Skeletonization')
plt.show()
