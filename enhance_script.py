import matplotlib.pyplot as plt
import cv2
from enhance import enhance_medical_image
import numpy as np

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

img = (cv2.imread('imagenes_orginales/Caso A BN.png'))
img_gray = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = img_gray[y:y + h, x:x + w]

enhanced = enhance_medical_image(zoom)
denoised_enhanced = cv2.fastNlMeansDenoising(enhanced.astype(np.uint8))

fig = plt.figure()
f1 = fig.add_subplot(1, 3, 1)
imgplot = plt.imshow(zoom, cmap="gray")
f1.set_title("Orginal")

f1 = fig.add_subplot(1, 3, 2)
imgplot = plt.imshow(enhanced, cmap="gray")
f1.set_title("Enhanced")

f1 = fig.add_subplot(1, 3, 3)
imgplot = plt.imshow(denoised_enhanced, cmap="gray")
f1.set_title("Enhanced + denoise")

plt.show()
