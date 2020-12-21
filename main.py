import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import time

from subpixel_edges import init, subpixel_edges
from anisotropic_difussion import anisodiff

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

if __name__ == '__main__':
    this_path = os.path.dirname(os.path.realpath(__file__))
    img = cv2.imread(os.path.join(this_path, 'imagenes_orginales/Caso A BN.png'))

    img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
    zoom = img_gray[y:y + h, x:x + w]

    after_aniso = anisodiff(zoom)

    iters = 2
    threshold = 1.5
    order = 2
    print("Initializing...")
    now = time()
    init()
    elapsed = time() - now
    print("Initialization time: ", elapsed)

    fig = plt.figure()

    print("Processing original image...")
    now = time()
    edges = subpixel_edges(zoom, threshold, iters, order)
    elapsed = time() - now
    print("Processing time: ", elapsed)

    f1 = fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(zoom, cmap="gray")
    imgplot = plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
    f1.set_title("Orginal")

    print("Processing image with anisotropic diffusion...")
    now = time()
    edges = subpixel_edges(after_aniso, threshold, iters, order)
    elapsed = time() - now
    print("Processing time: ", elapsed)

    f1 = fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(after_aniso, cmap="gray")
    imgplot = plt.quiver(edges.x, edges.y, edges.nx, -edges.ny, scale=40)
    f1.set_title("Anisotropic difussion")

    plt.show()
