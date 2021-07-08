import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from VeinSegmentation.Enhance import enhanceMedicalImage
from VeinSegmentation.Skeletonization import skeletonization

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

mpl.rcParams['figure.dpi'] = 500

iters = 2
threshold = 4.5
order = 2

original = (cv2.imread('../imagenes_orginales/Caso A BN.png'))
zoom = original[y:y + h, x:x + w]

enhanced = enhanceMedicalImage(zoom.copy())

old = skeletonization(enhanced.copy())
print("{}".format(sum(sum(old == 255))))
plt.imshow(old, cmap='gray')
plt.title('Old Skeletonization')
plt.show()

h = old.shape[0]
w = old.shape[1]

new = np.zeros((h, w))
# iteramos sobre cada pixel
for idx in range(0, 10):
    for i in range(0, h):
        for j in range(0, w):
            region = old[j:j + 3, i:i + 3]
            if sum(sum(region == 255)) == 3:
                new[j:j + 3, i:i + 3] = region

    print("{}".format(sum(sum(new == 255))))

plt.imshow(new, cmap='gray')
plt.title('Proccessed Skeletonization')
plt.show()
