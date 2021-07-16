import tkinter as tk

import cv2

from src.Utils.Utils import openCVToPIL, PILtoOpenCV
from src.VeinSegmentation import Enhance
from src.VeinSegmentation.Processing import Processing


class BrightnessContrastDialog:
    def __init__(self, parent, pil_image):
        self.parent = parent.children['!app']
        self.pil_image = pil_image
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        title = "Contraste y brillo"
        self.top.title(title)

        brightnessLabelText = "Seleccione un valor para el brillo"
        tk.Label(self.top, text=brightnessLabelText).pack()
        self.brightnessSlider = tk.Scale(self.top, length=200, from_=-255, to=255, orient=tk.HORIZONTAL)
        self.brightnessSlider.set(self.parent.brightnessValue)
        self.brightnessSlider.configure(command=self.brightAndContrastController)
        self.brightnessSlider.pack()
        print('Initial brightness value {}'.format(self.parent.brightnessValue))

        contrastLabelText = "Seleccione un valor para el contraste"
        tk.Label(self.top, text=contrastLabelText).pack()
        self.contrastSlider = tk.Scale(self.top, length=200, from_=-127, to=127, orient=tk.HORIZONTAL)
        self.contrastSlider.set(self.parent.contrastValue)
        self.contrastSlider.configure(command=self.brightAndContrastController)
        self.contrastSlider.pack()
        print('Initial contrast value {}'.format(self.parent.contrastValue))

        self.brightAndContrastController()

        self.button = tk.Button(self.top, text="Reiniciar", command=self.reset)
        self.button.pack()

    def reset(self):
        print('BrightnessContrastDialog::reset')
        self.contrastSlider.set(0)
        self.brightnessSlider.set(0)

        self.parent.brightnessValue = 0
        self.parent.contrastValue = 0

        self.brightAndContrastController()

    def brightAndContrastController(self, event=None, reset=False):
        brightness = self.brightnessSlider.get()
        contrast = self.contrastSlider.get()

        print('Controler values b={} c={}'.format(brightness, contrast))

        img = PILtoOpenCV(self.pil_image.copy())

        self.parent.brightnessValue = brightness
        self.parent.contrastValue = contrast

        if len(self.parent.polygonPoints) > 1:
            processing = Processing(img, self.parent.mask)
            result = processing.brightnessAndContrast(brightness, contrast)
        else:
            result = Enhance.processBrightnessAndContrast(img, brightness, contrast)

        self.parent.openCVImage = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
        self.parent.image = openCVToPIL(result)
        self.parent.showImage()

