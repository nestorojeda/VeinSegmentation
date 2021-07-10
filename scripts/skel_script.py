import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams['figure.dpi'] = 900

iters = 2
threshold = 4.5
order = 2

skeleton = (cv2.imread('../skeleton.png'))
skeleton = cv2.cvtColor(skeleton, cv2.COLOR_RGB2GRAY)
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
