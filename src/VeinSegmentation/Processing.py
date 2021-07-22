import cv2
import matplotlib.pyplot as plt
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
        self.skeleton = None
        self.subpixelImage = None
        self.transparentSkeleton = None
        self.whiteSkeleton = None

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
        if self.skeleton is not None:
            return self.mergeCropAndOriginal(self.skeleton.copy())
        crop = self.getROICrop()

        if self.enhanced is None:
            crop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)
            self.enhanced = crop.copy()
        else:
            crop = self.enhanced.copy()

        skelCrop, self.skeletonContours = skeletonization(crop)
        self.whiteSkeleton = cleanSkeleton(skelCrop)
        skelCrop = cv2.cvtColor(skelCrop.astype(np.uint8), cv2.COLOR_GRAY2RGB)

        skelCrop[np.where((skelCrop[:, :, 2] == 255))] = (0, 0, 255)

        self.skeleton = skelCrop

        openContours, closedContours = Contour.sortContours(self.skeletonContours)
        result = cv2.drawContours(skelCrop.copy(), closedContours, -1, (0, 255, 0), 1)

        return self.mergeCropAndOriginal(result)

    def skeletonSettings(self, centerLine, contour, transparency, pixelWidth=1):
        crop = self.getROICrop()
        if centerLine:
            result = self.skeleton.copy()
        else:
            result = np.zeros(crop.shape)

        if transparency:
            if centerLine:
                if self.transparentSkeleton is None:
                    result[np.where((self.skeleton[:, :, 2] != 255))] = crop[np.where((self.skeleton[:, :, 2] != 255))]
                    self.transparentSkeleton = result.copy()
                else:
                    result = self.transparentSkeleton.copy()
            else:
                result = crop.copy()

        if contour:
            openContours, closedContours = Contour.sortContours(self.skeletonContours)
            result = cv2.drawContours(result, closedContours, -1, (0, 255, 0), pixelWidth)

        return self.mergeCropAndOriginal(result)

    def getCleanedSkeleton(self):
        return self.whiteSkeleton

    def enhance(self):
        if self.enhanced is not None:
            return self.mergeCropAndOriginal(self.enhanced)
        crop = self.getROICrop()
        enhancedCrop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)  # Corte mejorado
        self.enhanced = enhancedCrop.copy()

        return self.mergeCropAndOriginal(enhancedCrop)

    def subpixel(self, iters=2, threshold=1.5, order=2):
        if self.subpixelImage is not None:
            return self.mergeCropAndOriginal(self.subpixelImage.copy())
        crop = self.getROICrop()
        if self.enhanced is None:
            crop = Enhance.enhanceMedicalImage(crop).astype(np.uint8)
            self.enhanced = crop.copy()
        else:
            crop = self.enhanced.copy()

        if len(crop.shape) != 2:
            crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

        edges = subpixel_edges(crop.astype(float), threshold, iters, order)
        edgedCrop = cv2.cvtColor(crop.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        self.subpixelImage = edgedCrop

        for point in np.array((edges.x, edges.y)).T.astype(np.uint):
            cv2.circle(edgedCrop, tuple(point), 1, (0, 0, 255))
        return self.mergeCropAndOriginal(edgedCrop)

    def brightnessAndContrast(self, brightness, contrast):
        crop = self.getROICrop()
        modifiedCrop = Enhance.processBrightnessAndContrast(crop, brightness, contrast)
        return self.mergeCropAndOriginal(modifiedCrop)

    def mergeCropAndOriginal(self, processedCrop):
        if len(processedCrop.shape) != 3:
            processedCrop = cv2.cvtColor(processedCrop, cv2.COLOR_GRAY2RGB)
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

    def correctSkeleton(self, points):
        x, y, w, h = self.getCropCoordinates()
        pointCrop = [(points[0][0] - x, points[0][1] - y), (points[1][0] - x, points[1][1] - y)]

        self.whiteSkeleton = cv2.line(self.whiteSkeleton,
                                      pointCrop[0], pointCrop[1],
                                      color=255,
                                      thickness=1)
        self.skeleton = cv2.line(self.skeleton,
                                 pointCrop[0], pointCrop[1],
                                 color=(0, 0, 255),
                                 thickness=1)
