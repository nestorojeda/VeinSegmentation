import math
import tkinter as tk
from tkinter import messagebox

from VeinSegmentation import Mask
from VeinSegmentation.Skeletonization import skeletonLenght


class VeinMetricsModal:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        title = "Métricas"
        self.top.title(title)

        skeleton = self.parent.cleanedSkeleton
        area = Mask.getMaskArea(self.parent.mask)
        isSkeletonized = self.parent.isSkeletonized
        pixelSize = self.parent.pixelSize
        squarePixelSize = (1 / pixelSize) ** 2
        print('Area in pixels is {} px'.format(area))
        if pixelSize:
            tk.Label(self.top, text="Area de la selección: {} cm2".format(area / squarePixelSize)).pack()
        if isSkeletonized and pixelSize:
            measure = skeletonLenght(skeleton, pixelSize)
            tk.Label(self.top, text="Longitud de la red venosa: {} cm".format(measure)).pack()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.top.destroy()
