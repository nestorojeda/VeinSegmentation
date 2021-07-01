import math
import tkinter as tk
from tkinter import messagebox

import cv2
import matplotlib.pyplot as plt

from Utils.Utils import openCVToPIL, PILtoOpenCV


class ReferencePointsDialog:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        title = "Referencia de la medida"
        self.top.title(title)

        vcmd = (self.top.register(self.validate),
                '%d', '%i', '%P', '%s', '%S', '%v', '%V', '%W')

        if len(self.parent.reference_points) >= 1:
            first_point_label = self.parent.reference_points[0]
        else:
            first_point_label = '-'

        if len(self.parent.reference_points) >= 2:
            second_point_label = self.parent.reference_points[1]
        else:
            second_point_label = '-'

        tk.Label(self.top, text="Posición del primer punto: {}".format(first_point_label)).pack()
        tk.Label(self.top, text="Posición del segunto punto: {}".format(second_point_label)).pack()
        tk.Label(self.top, text="Introduce la medida en centímetros").pack()
        self.entry = tk.Entry(self.top, validate='key', validatecommand=vcmd)
        self.entry.pack()
        self.button = tk.Button(self.top, text="Aceptar", command=self.saveReference)
        self.button.pack()

    def validate(self, action, index, value_if_allowed,
                 prior_value, text, validation_type, trigger_type, widget_name):
        # https://stackoverflow.com/questions/8959815/restricting-the-value-in-tkinter-entry-widget
        if value_if_allowed:
            try:
                float(value_if_allowed)
                return True
            except ValueError:
                return False
        else:
            return False

    def saveReference(self):
        print('ReferencePointsDialog::saveReference')
        pixel_distance = math.sqrt((self.parent.reference_points[1][0] - self.parent.reference_points[0][0]) ** 2 +
                                   (self.parent.reference_points[1][1] - self.parent.reference_points[0][1]) ** 2)
        print('Pixel distance {}'.format(pixel_distance))
        self.parent.one_pixel_size = (float(self.entry.get())) / pixel_distance
        print('Real pixel size {}'.format(self.parent.one_pixel_size))
        messagebox.showinfo('Aviso', 'Cada pixel de su imagen corresponde a {} cm.'.format(self.parent.one_pixel_size))
        self.cancel()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.top.destroy()
