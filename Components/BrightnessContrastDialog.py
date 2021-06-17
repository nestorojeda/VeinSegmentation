import tkinter as tk
import cv2
import matplotlib.pyplot as plt

from Utils.Utils import openCVToPIL, PILtoOpenCV
from VeinSegmentation import Enhance


class BrightnessContrastDialog:
    def __init__(self, parent, pil_image):
        self.parent = parent.children['!app']
        self.pil_image = pil_image
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        # self.top.grab_set()
        title = "Contraste y brillo"
        self.top.title(title)

        brightness_labeltext = "Seleccione un valor para el brillo"
        tk.Label(self.top, text=brightness_labeltext).pack()
        self.b_slider = tk.Scale(self.top, length=200, from_=-255, to=255, orient=tk.HORIZONTAL)
        self.b_slider.set(self.parent.brightness_value)
        self.b_slider.configure(command=self.bright_and_contrast_controller)
        self.b_slider.pack()
        print('Initial brightness value {}'.format(self.parent.brightness_value))

        contrast_labeltext = "Seleccione un valor para el contraste"
        tk.Label(self.top, text=contrast_labeltext).pack()
        self.c_slider = tk.Scale(self.top, length=200, from_=-127, to=127, orient=tk.HORIZONTAL)
        self.c_slider.set(self.parent.contrast_value)
        self.c_slider.configure(command=self.bright_and_contrast_controller)
        self.c_slider.pack()
        print('Initial contrast value {}'.format(self.parent.contrast_value))

        self.bright_and_contrast_controller()

        self.button = tk.Button(self.top, text="Reiniciar", command=self.reset)
        self.button.pack()

    def reset(self):
        print('BrightnessContrastDialog::reset')
        self.c_slider.set(0)
        self.b_slider.set(0)

        self.parent.brightness_value = 0
        self.parent.contrast_value = 0

        self.bright_and_contrast_controller()

        self.parent.image = self.pil_image
        self.parent.show_image()

    def bright_and_contrast_controller(self, event=None, reset=False):
        brightness = self.b_slider.get()
        contrast = self.c_slider.get()

        print('Controler values b={} c={}'.format(brightness, contrast))

        img = PILtoOpenCV(self.pil_image.copy())

        self.parent.brightness_value = brightness
        self.parent.contrast_value = contrast

        buf = Enhance.process_brightness_and_contrast(brightness, contrast, img)

        self.parent.image = openCVToPIL(buf)
        self.parent.show_image()

