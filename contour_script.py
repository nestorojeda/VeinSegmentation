import cv2
import imutils as imutils
import matplotlib.pyplot as plt

image = (cv2.imread('imagenes_orginales/Caso A BN.png'))
gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
ret, thresh = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(gray, contours, -1, (0, 255, 0), 3)

plt.imshow(gray, cmap='gray')
plt.show()

ratio = image.shape[0] / 300.0
orig = image.copy()
image = imutils.resize(image, height = 300)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)
edged = cv2.Canny(gray, 30, 200)

plt.imshow(edged, cmap='gray')
plt.show()