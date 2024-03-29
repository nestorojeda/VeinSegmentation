import tkinter as tk

from src.VeinSegmentation.Skeletonization import skeletonLenght


class VeinMetricsModal:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        title = "Métricas"
        self.top.title(title)

        x = self.parent.winfo_x()
        y = self.parent.winfo_y()
        self.top.geometry("+%d+%d" % (x + 300, y + 200))

        area = self.parent.processing.getMaskArea()
        isSkeletonized = self.parent.isSkeletonized
        pixelSize = self.parent.pixelSize
        print('Area in pixels is {} px'.format(area))
        if pixelSize:
            squarePixelSize = (1 / pixelSize) ** 2
            tk.Label(self.top, text="Area de la selección: {} cm2".format(area / squarePixelSize)).pack()
        if isSkeletonized and pixelSize:
            measure = skeletonLenght(self.parent.processing.getCleanedSkeleton(), pixelSize)
            tk.Label(self.top, text="Longitud de la red venosa: {} cm".format(measure)).pack()
        if not pixelSize and not isSkeletonized:
            tk.Label(self.top, text="Debes seleccionar una referencia para poder obtener información").pack()

    def cancel(self):
        self.parent.focus_set()
        self.top.destroy()
