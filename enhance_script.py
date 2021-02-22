import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from enhance import enhance_medical_image, km_clust, skeletonization

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 4.5
order = 2

original = (cv2.imread('imagenes_orginales/Caso A BN.png'))
enhanced_gray = (cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = enhanced_gray[y:y + h, x:x + w]

pictures = []  # dictionary with all generated pictures

enhanced = enhance_medical_image(zoom)

# Group similar grey levels using 8 clusters
values, labels = km_clust(enhanced, n_clusters=2)
# Create the segmented array from labels and values
enhanced_segm = np.choose(labels, values)
# Reshape the array as the original image
enhanced_segm.shape = enhanced.shape
# Get the values of min and max intensity in the original image
vmin = enhanced.min()
vmax = enhanced.max()
fig = plt.figure(1)
# Plot the original image
ax1 = fig.add_subplot(1, 2, 1)
ax1.imshow(enhanced, cmap=plt.cm.gray)
ax1.set_title('Original image')
# Plot the simplified color image
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(enhanced_segm, cmap=plt.cm.gray)
ax2.set_title('Simplified levels')
# Get rid of the tick labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.show()

skeletonization(enhanced_segm, 1000)

# pictures.append(Picture(zoom, "Original"))
# pictures.append(Picture(enhanced, "Enhanced"))
# pictures.append(Picture(quantized, "Cuantificada"))
# show(pictures)

# edges = subpixel_edges(enhanced.astype(float), threshold, iters, order)
# plt.imshow(enhanced, cmap="gray")
# plt.plot(edges.x, edges.y,  'ro', markersize=.1)
# plt.show()
