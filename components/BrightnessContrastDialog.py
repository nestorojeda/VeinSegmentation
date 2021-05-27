import tkinter as tk
import cv2


def controller(img, brightness=255,
               contrast=127):
    brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255))

    contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127))

    if brightness != 0:

        if brightness > 0:

            shadow = brightness

            max = 255

        else:

            shadow = 0
            max = 255 + brightness

        al_pha = (max - shadow) / 255
        ga_mma = shadow

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(img, al_pha,
                              img, 0, ga_mma)

    else:
        cal = img

    if contrast != 0:
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)

        # The function addWeighted calculates
        # the weighted sum of two arrays
        cal = cv2.addWeighted(cal, Alpha,
                              cal, 0, Gamma)

    # putText renders the specified text string in the image.
    cv2.putText(cal, 'B:{},C:{}'.format(brightness,
                                        contrast), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return cal


class BrightnessContrastDialog:
    def __init__(self, parent):
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        #self.top.grab_set()
        title = "Contraste y brillo"
        contrast_labeltext = "Seleccione un valor para el contraste"
        brightness_labeltext = "Seleccione un valor para el brillo"
        self.top.title(title)

        tk.Label(self.top, text=contrast_labeltext).pack()
        b_slider = tk.Scale(self.top, length=200, from_=-255, to=255, orient=tk.HORIZONTAL)
        b_slider.set(0)
        b_slider.pack()

        tk.Label(self.top, text=brightness_labeltext).pack()
        c_slider = tk.Scale(self.top, length=200, from_=-127, to=127, orient=tk.HORIZONTAL)

        c_slider.set(0)
        c_slider.pack()
        self.top.bind("<Return>", self.ok)
        b = tk.Button(self.top, text="OK", command=self.ok)
        b.pack(pady=5)

    def ok(self, event=None):
        print('hola')
        self.top.destroy()

    def cancel(self, event=None):
        self.top.destroy()
