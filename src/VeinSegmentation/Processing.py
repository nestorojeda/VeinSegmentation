import cv2
import numpy as np
from subpixel_edges import subpixel_edges

from src.VeinSegmentation import Enhance, Contour
from src.VeinSegmentation.Skeletonization import skeletonization, cleanSkeleton


class Processing:

    def __init__(self, image, mask):
        self.image = image
        self.mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        self.crops = []
        self.enhanced = None
        self.transparentSkeleton = None
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
        if self.enhanced is None:
            crop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)
            self.enhanced = crop.copy()
        else:
            crop = self.enhanced.copy()

        skelCrop, self.skeletonContours = skeletonization(crop)
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
            if self.transparentSkeleton is None:
                for j in range(0, self.skeleton.shape[0]):
                    for i in range(0, self.skeleton.shape[1]):
                        if np.array_equal(self.skeleton[j, i], np.array([0, 0, 0])):
                            result[j, i] = crop[j, i]
                self.transparentSkeleton = result.copy()
            else:
                result = self.transparentSkeleton.copy()

        if contour:
            openContours, closedContours = Contour.sortContours(self.skeletonContours)
            result = cv2.drawContours(result, closedContours, -1, (0, 255, 0), pixelWidth)

        return self.mergeCropAndOriginal(result)

    def getCleanedSkeleton(self):
        return cleanSkeleton(self.whiteSkeleton)

    def enhance(self):
        crop = self.getROICrop()
        enhancedCrop = Enhance.enhanceMedicalImage(crop)  # Corte mejorado
        self.enhanced = enhancedCrop.astype(np.uint8)
        enhancedCrop = cv2.cvtColor(enhancedCrop.astype(np.uint8).copy(), cv2.COLOR_GRAY2RGB)

        return self.mergeCropAndOriginal(enhancedCrop)

    def subpixel(self, iters=2, threshold=1.5, order=2):
        crop = self.getROICrop()

        if self.enhanced is None:
            crop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)
            self.enhanced = crop
        else:
            crop = self.enhanced

        edges = subpixel_edges(crop.astype(float), threshold, iters, order)

        edgedCrop = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        for point in np.array((edges.x, edges.y)).T.astype(np.uint):
            cv2.circle(edgedCrop, tuple(point), 1, (0, 0, 255))
        return self.mergeCropAndOriginal(edgedCrop)

    def brightnessAndContrast(self, brightness, contrast):
        crop = self.getROICrop()
        modifiedCrop = Enhance.processBrightnessAndContrast(crop, brightness, contrast)
        return self.mergeCropAndOriginal(modifiedCrop)

    def mergeCropAndOriginal(self, processedCrop):
        merged = self.image.copy()
        x, y, w, h = self.getCropCoordinates()
        merged[y:y + h, x:x + w] = processedCrop
        fg = cv2.bitwise_or(merged, merged, mask=self.mask)
        invertedMask = cv2.bitwise_not(self.mask)
        fgbg = cv2.bitwise_or(fg, self.image, mask=invertedMask)
        result = cv2.bitwise_or(fgbg, fg)
        return result

    def getMaskArea(self):
        """ Obtención del numero de pixeles de la imagen"""
        contours, hierarchy = cv2.findContours(self.mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        area = 0
        for cnt in contours:
            area += cv2.contourArea(cnt)

        return area