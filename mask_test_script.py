import cv2
import matplotlib.pyplot as plt
from VeinSegmentation import mask as mk

image = cv2.imread('imagenes_orginales/Caso A BN.png')
mask = cv2.imread('./mask.png')

enhanced_roi = mk.apply_enhance_to_roi(image, mask)
plt.imshow(enhanced_roi, cmap='gray')
plt.title('enhance roi')
plt.show()

subpixel_roi = mk.apply_subpixel_to_roi(enhanced_roi.astype(float), mask)
plt.imshow(subpixel_roi, cmap='gray')
plt.title('subpixel roi')
plt.show()

skeletonized_roi = mk.apply_skeletonization_to_roi(enhanced_roi, mask)
plt.imshow(skeletonized_roi, cmap='gray')
plt.title('skeletonized roi')
plt.show()
