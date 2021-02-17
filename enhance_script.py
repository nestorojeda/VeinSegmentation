from datetime import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pictures import show, Picture
from subpixel_edges import subpixel_edges
from enhance import enhance_medical_image
from sklearn import cluster
from skimage import data

scaleX = 1
scaleY = 1

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y

iters = 2
threshold = 4.5
order = 2


def km_clust(array, n_clusters):
    # Create a line array, the lazy way
    X = array.reshape((-1, 1))
    # Define the k-means clustering problem
    k_m = cluster.KMeans(n_clusters=n_clusters, n_init=4)
    # Solve the k-means clustering problem
    k_m.fit(X)
    # Get the coordinates of the clusters centres as a 1D array
    values = k_m.cluster_centers_.squeeze()
    # Get the label of each point
    labels = k_m.labels_
    return values, labels


original = (cv2.imread('imagenes_orginales/Caso A BN.png'))
enhanced_gray = (cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)).astype(float)
zoom = enhanced_gray[y:y + h, x:x + w]

pictures = []  # dictionary with all generated pictures

enhanced = enhance_medical_image(zoom)

# Group similar grey levels using 8 clusters
values, labels = km_clust(enhanced, n_clusters=4)
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
ax1.imshow(enhanced, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax1.set_title('Original image')
# Plot the simplified color image
ax2 = fig.add_subplot(1, 2, 2)
ax2.imshow(enhanced_segm, cmap=plt.cm.gray, vmin=vmin, vmax=vmax)
ax2.set_title('Simplified levels')
# Get rid of the tick labels
ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticklabels([])
plt.show()

# pictures.append(Picture(zoom, "Original"))
# pictures.append(Picture(enhanced, "Enhanced"))
# pictures.append(Picture(quantized, "Cuantificada"))
# show(pictures)

# edges = subpixel_edges(enhanced.astype(float), threshold, iters, order)
# plt.imshow(enhanced, cmap="gray")
# plt.plot(edges.x, edges.y,  'ro', markersize=.1)
# plt.show()
