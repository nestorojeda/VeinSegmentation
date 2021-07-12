import os

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from src.VeinSegmentation import Skeletonization, Enhance

mpl.rcParams['figure.dpi'] = 900

iters = 2
threshold = 4.5
order = 2

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

this_path = os.path.dirname(os.path.realpath(__file__))
img = cv2.imread(os.path.join(this_path, '../imagenes_orginales/Caso A BN.png'))

img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = img_gray[y:y + h, x:x + w]

enhanced = Enhance.enhanceMedicalImage(zoom)
skeleton = Skeletonization.skeletonization(enhanced)

# img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
# skeleton = (cv2.imread('../skeleton.png'))
# skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
plt.imshow(skeleton)
plt.title('original')
plt.show()

h = skeleton.shape[0]
w = skeleton.shape[1]

pixelSize = 0.02
measure = 0

# Clean blobs
cleanSkeleton = np.zeros((h, w))
for j in range(0, h):
    for i in range(0, w):
        region = skeleton[j:j + 3, i:i + 3]
        if sum(sum(region == 255)) == 3:
            cleanSkeleton[j:j + 3, i:i + 3] = region

plt.imshow(cleanSkeleton)
plt.title('cleanSkeleton')
plt.show()

skeleton = cleanSkeleton.copy()

# Clean edge diagonal pixels
for i in range(0, h):
    for j in range(0, w):
        if skeleton[i, j] == 255:
            if (i + 1) < h and (j + 1) < w:
                if skeleton[i + 1, j + 1] == 255:
                    if skeleton[i + 1, j] == 255:
                        skeleton[i + 1, j] = 0
                    if skeleton[i, j + 1] == 255:
                        skeleton[i, j + 1] = 0

            if (i + 1) < h and (j - 1) >= 0:
                if skeleton[i + 1, j - 1] == 255:
                    if skeleton[i + 1, j] == 255:
                        skeleton[i + 1, j] = 0
                    if skeleton[i, j - 1] == 255:
                        skeleton[i, j - 1] = 0

plt.imshow(skeleton)
plt.title('cleanSkeleton 2')
plt.show()

# Measure
for i in range(0, h):
    for j in range(0, w):
        if skeleton[i, j] == 255:
            # Diagonal
            if (i + 1) < h and (j + 1) < w:
                if skeleton[i + 1, j + 1] == 255:
                    measure += np.sqrt(2) * pixelSize
            # Vertical
            if (i + 1) < h:
                if skeleton[i + 1, j] == 255:
                    measure += pixelSize
            # Horizontal
            if (j + 1) < w:
                if skeleton[i, j + 1] == 255:
                    measure += pixelSize

print("Medida = {} cm".format(measure))
