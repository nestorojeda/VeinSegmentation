import tkinter

import PIL.Image
import PIL.ImageTk
import cv2

from VeinSegmentation.enhance import enhance_medical_image

y = 1300  # donde empieza el corte en y
x = 1600  # donde empieza el corte en x
h = 600  # tamaño del corte en h
w = 600  # tamaño del corte en y


class App:
    def __init__(self, window, window_title, image_path="imagenes_orginales/Caso A BN.png"):
        self.window = window
        self._job = None
        self.window.title(window_title)

        self.clip_limit = 5
        self.tile_grid_size = 5

        # Load an image using OpenCV
        self.cv_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        self.cv_img = self.cv_img[y:y + h, x:x + w]
        self.cv_img_original = self.cv_img

        # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
        self.height, self.width, no_channels = self.cv_img.shape

        # Create a canvas that can fit the above image
        self.canvas = tkinter.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))

        # Add a PhotoImage to the Canvas
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

        self.clipLimitSlider = tkinter.Scale(window,
                                             from_=0,
                                             to=255,
                                             length=600,
                                             tickinterval=32,
                                             orient=tkinter.HORIZONTAL,
                                             label="ClipLimit")
        self.clipLimitSlider.bind("<ButtonRelease-1>", self.updateClipLimit)
        self.clipLimitSlider.set(23)
        self.clipLimitSlider.pack()

        self.tileGridSizeSlider = tkinter.Scale(window,
                                                from_=0,
                                                to=50,
                                                length=600,
                                                tickinterval=5,
                                                orient=tkinter.HORIZONTAL,
                                                label="TileGridSize")
        self.tileGridSizeSlider.bind("<ButtonRelease-1>", self.updateGridSize)
        self.tileGridSizeSlider.set(23)
        self.tileGridSizeSlider.pack()

        self.btn_revert = tkinter.Button(window, text="Revert to original", width=50, command=self.revert_to_original)
        self.btn_revert.pack(anchor=tkinter.CENTER, expand=True)

        self.window.mainloop()

    def updateClipLimit(self, event):
        if self._job:
            self.window.after_cancel(self._job)

        self.clip_limit = self.clipLimitSlider.get()
        self._job = self.window.after(500, self.updateImage)

    def updateGridSize(self, event):
        if self._job:
            self.window.after_cancel(self._job)

        self.tile_grid_size = self.tileGridSizeSlider.get()
        self._job = self.window.after(500, self.updateImage)

    def updateImage(self):
        self.cv_img = enhance_medical_image(self.cv_img_original, clip_limit=self.clip_limit,
                                            tile_grid_size=self.tile_grid_size)
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

    def revert_to_original(self):
        self.cv_img = self.cv_img_original
        self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(self.cv_img))
        self.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)


# Create a window and pass it to the Application object
App(tkinter.Tk(), "Test")
