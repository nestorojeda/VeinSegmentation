import cv2
import numpy as np
from VeinSegmentation.Enhance import enhanceMedicalImage, km_clust, skeletonization

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
zoom = enhanced_gray[y:y + h, x:x + w]

pictures = []  # dictionary with all generated pictures

enhanced = enhanceMedicalImage(zoom)

# Group similar grey levels using 8 clusters
values, labels = km_clust(enhanced.astype('float32'), n_clusters=20)
# Create the segmented array from labels and values
enhanced_segm = np.choose(labels, values)
# Reshape the array as the original image
enhanced_segm.shape = enhanced.shape
# Get the values of min and max intensity in the original image
vmin = enhanced.min()
vmax = enhanced.max()

# Realizo el umbralziado de la imagen antes de invertirla para que la esqueletonizacion haga el proceso sobre las
# lineas blancas
ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
inverted_segm_th = cv2.bitwise_not(img)

skel = skeletonization(inverted_segm_th, 100)

pictures.append(Picture(zoom, "Original"))
pictures.append(Picture(skel, "Esqueleto"))
show(pictures)

# edges = subpixel_edges(enhanced.astype(float), threshold, iters, order)
# plt.imshow(enhanced, cmap="gray")
# plt.plot(edges.x, edges.y,  'ro', markersize=.1)
# plt.show()