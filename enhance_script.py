from datetime import time

import matplotlib.pyplot as plt
import cv2
from enhance import enhance_medical_image
import numpy as np
from pictures import show, Picture
from subpixel_edges import subpixel_edges

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 4.5
order = 2

img = (cv2.imread('imagenes_orginales/Caso A BN.png'))
img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = img_gray[y:y + h, x:x + w]

pictures = []  # dictionary with all generated pictures

enhanced = enhance_medical_image(zoom)
denoised_enhanced = cv2.fastNlMeansDenoising(enhanced.astype(np.uint8))
th, im_gray_th_otsu = cv2.threshold(denoised_enhanced, 128, 255, cv2.THRESH_OTSU)

pictures.append(Picture(zoom, "Original"))
pictures.append(Picture(enhanced, "Enhanced"))
pictures.append(Picture(denoised_enhanced, "Enhanced + denoise"))
pictures.append(Picture(im_gray_th_otsu, "Enhanced + denoise + threshold"))

edges = subpixel_edges(denoised_enhanced.astype(float), threshold, iters, order)

plt.imshow(denoised_enhanced, cmap="gray")
plt.plot(edges.x, edges.y,  'ro', markersize=.1)
plt.show()

show(pictures)
