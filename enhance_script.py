from datetime import time

import matplotlib.pyplot as plt
import cv2
from enhance import enhance_medical_image
import numpy as np
from pictures import show, Picture
from subpixel_edges import subpixel_edges

from utils import OpenCVToPIL

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 4.5
order = 2
levels = 20

NCLUSTERS = 8
NROUNDS = 1

img = (cv2.imread('imagenes_orginales/Caso A BN.png'))
img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = img_gray[y:y + h, x:x + w]

pictures = []  # dictionary with all generated pictures

enhanced = enhance_medical_image(zoom)
quantized = OpenCVToPIL(enhanced.astype(np.uint8)).quantize(64)


pictures.append(Picture(zoom, "Original"))
pictures.append(Picture(enhanced, "Enhanced"))
pictures.append(Picture(quantized, "Cuantificada"))
show(pictures)

# edges = subpixel_edges(enhanced.astype(float), threshold, iters, order)
# plt.imshow(enhanced, cmap="gray")
# plt.plot(edges.x, edges.y,  'ro', markersize=.1)
# plt.show()
