import math
import tkinter as tk
from tkinter import messagebox

from VeinSegmentation import Mask


class VeinMetricsModal:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        title = "Métricas"
        self.top.title(title)

        white_pixels = self.parent.white_pixels
        black_pixels = self.parent.black_pixels
        area = Mask.get_mask_area(self.parent.mask)
        is_skeletonized = self.parent.is_skeletonized
        is_enhanced = self.parent.is_enhanced
        pixel_size = self.parent.one_pixel_size
        square_pixel_size = (1 / pixel_size) ** 2
        print('Area in pixels is {} px'.format(area))
        if pixel_size:
            tk.Label(self.top, text="Area de la selección: {} cm2".format(area / square_pixel_size)).pack()
        if is_skeletonized and pixel_size and white_pixels:
            tk.Label(self.top, text="Longitud de la red venosa: {} cm".format(pixel_size * white_pixels)).pack()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.top.destroy()
