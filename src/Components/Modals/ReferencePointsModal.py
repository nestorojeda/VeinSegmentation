import math
import tkinter as tk
from tkinter import messagebox


class ReferencePointsDialog:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = tk.Toplevel(parent)
        self.top.transient(parent)
        self.top.grab_set()
        title = "Referencia de la medida"
        self.top.title(title)

        x = self.parent.winfo_x()
        y = self.parent.winfo_y()
        self.top.geometry("+%d+%d" % (x + 300, y + 200))



        if len(self.parent.referencePoints) >= 1:
            firstPointLabel = self.parent.referencePoints[0]
        else:
            firstPointLabel = '-'

        if len(self.parent.referencePoints) >= 2:
            secondPointLabel = self.parent.referencePoints[1]
        else:
            secondPointLabel = '-'

        tk.Label(self.top, text="Posición del primer punto: {}".format(firstPointLabel)).pack()
        tk.Label(self.top, text="Posición del segunto punto: {}".format(secondPointLabel)).pack()
        tk.Label(self.top, text="Introduce la medida en centímetros").pack()
        vcmd = (self.top.register(self.callback))
        self.entry = tk.Entry(self.top, validate='key', validatecommand=(vcmd, '%P'))
        self.entry.pack()
        self.button = tk.Button(self.top, text="Aceptar", command=self.saveReference)
        self.button.pack()

    def callback(self, P):
        if str.isdigit(P) or P == "":
            return True
        else:
            return False

    def saveReference(self):
        print('ReferencePointsDialog::saveReference')
        pixel_distance = math.sqrt((self.parent.referencePoints[1][0] - self.parent.referencePoints[0][0]) ** 2 +
                                   (self.parent.referencePoints[1][1] - self.parent.referencePoints[0][1]) ** 2)
        print('Pixel distance {}'.format(pixel_distance))
        self.parent.pixelSize = (float(self.entry.get())) / pixel_distance
        print('Real pixel size {}'.format(self.parent.pixelSize))
        messagebox.showinfo('Aviso', 'Cada pixel de su imagen corresponde a {} cm.'.format(self.parent.pixelSize))
        self.cancel()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.top.destroy()
