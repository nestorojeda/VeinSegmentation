import cv2
import numpy as np

from src.VeinSegmentation import Enhance, Contour
from src.VeinSegmentation.Skeletonization import skeletonization, cleanSkeleton


class Processing:

    def __int__(self, image):
        self.image = image

    def __init__(self, image, mask):
        self.image = image
        self.mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        self.crops = []

        self.getROICrop()

    def setMask(self, mask):
        self.mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        self.crops = []

        self.getROICrop()

    def getROICrop(self):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx = 0
        for cnt in contours:
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            return self.image[y:y + h, x:x + w]

    def getCropCoordinates(self):
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        idx = 0
        for cnt in contours:
            idx += 1
            x, y, w, h = cv2.boundingRect(cnt)
            return x, y, w, h

    def skeletonization(self):
        crop = self.getROICrop()
        enhancedCrop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)

        skelCrop, self.skeletonContours = skeletonization(enhancedCrop)
        self.whiteSkeleton = cleanSkeleton(skelCrop)
        skelCrop = cv2.cvtColor(skelCrop.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        for j in range(0, skelCrop.shape[0]):
            for i in range(0, skelCrop.shape[1]):
                if np.array_equal(skelCrop[j, i], np.array([255, 255, 255])):
                    skelCrop[j, i] = (0, 0, 255)

        self.skeleton = skelCrop
        return self.mergeCropAndOriginal(skelCrop)

    def skeletonSettings(self, contour, transparency, pixelWidth=1):
        crop = self.getROICrop()
        result = self.skeleton.copy()
        if transparency:
            for j in range(0, self.skeleton.shape[0]):
                for i in range(0, self.skeleton.shape[1]):
                    if np.array_equal(self.skeleton[j, i], np.array([0, 0, 0])):
                        result[j, i] = crop[j, i]
        if contour:
            openContours, closedContours = Contour.sortContours(self.skeletonContours)
            result = cv2.drawContours(result, closedContours, -1, (0, 255, 0), pixelWidth)

        return self.mergeCropAndOriginal(result)

    def getCleanedSkeleton(self):
        return cleanSkeleton(self.whiteSkeleton)

    def mergeCropAndOriginal(self, processedCrop):
        merged = self.image.copy()
        x, y, w, h = self.getCropCoordinates()
        merged[y:y + h, x:x + w] = processedCrop
        fg = cv2.bitwise_or(merged, merged, mask=self.mask)
        invertedMask = cv2.bitwise_not(self.mask)
        fgbg = cv2.bitwise_or(fg, self.image, mask=invertedMask)
        result = cv2.bitwise_or(fgbg, fg)
        return result
