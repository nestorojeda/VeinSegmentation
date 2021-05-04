import tkinter
import sys
import PIL.Image
import PIL.ImageTk
import cv2

from tkinter import Tk, Frame, Menu, messagebox
from VeinSegmentation.enhance import enhance_medical_image
from components.ResizableCanvas import ResizingCanvas

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y


# https://zetcode.com/tkinter/menustoolbars/
# https://solarianprogrammer.com/2018/04/20/python-opencv-show-image-tkinter-window/

class App(Frame):

    def __init__(self, window, image_path="imagenes_orginales/Caso A BN.png"):
        super().__init__(window)
        self._job = None
        self.window = window
        self.window.protocol("WM_DELETE_WINDOW", self.onExit)
        self.initUI(image_path)

    def initUI(self, image_path):

        menubar = Menu(self.master)
        self.master.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Exit", command=self.onExit)
        menubar.add_cascade(label="File", menu=fileMenu)

        # Load an image using OpenCV
        cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        height, width, no_channels = cv_img.shape

        # Create a canvas that can fit the above image
        canvas = ResizingCanvas(self.window, width=width, height=height)
        canvas.pack(fill="both", expand=True)

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(cv_img))

        # Add a PhotoImage to the Canvas
        canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

        self.window.mainloop()

    def onExit(self):
        if messagebox.askokcancel("Salir", "¿De seguro que quieres salir?"):
            self.window.destroy()
            sys.exit()


def main():
    root = Tk()
    root.title('Segmentación de venas')
    root.geometry("1000x500")
    app = App(root)
    app.mainloop()


if __name__ == '__main__':
    main()
