import cv2
import numpy as np
import matplotlib.pyplot as plt

from VeinSegmentation.enhance import segmentation

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
enhanced_gray = (cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)).astype(float)
img = enhanced_gray[y:y + h, x:x + w]

Z = img.reshape((-1, 1))
# convert to np.float32
Z = np.float32(Z)
# define criteria, number of clusters(K) and apply kmeans()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
# Now convert back into uint8, and make original image
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape(img.shape)

plt.imshow(res2, cmap='gray')
plt.title('OpenCV')
plt.show()

res3 = segmentation(img, n_clusters=3)
plt.imshow(res3, cmap='gray')
plt.title('Skimage')
plt.show()
