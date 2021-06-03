import sys
import tkinter as tk
from tkinter import Tk, Frame, messagebox, ttk, Menu
from tkinter import filedialog as fd

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk

import constants.colors as color
from components.BrightnessContrastDialog import BrightnessContrastDialog
from components.AutoScrollbar import AutoScrollbar
from utils.utils import openCVToPIL
from VeinSegmentation import mask

moving = True
drawing = False


# https://zetcode.com/tkinter/menustoolbars/
# https://solarianprogrammer.com/2018/04/20/python-opencv-show-image-tkinter-window/
# https://www.semicolonworld.com/question/55637/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width

class App(Frame):
    """ Advanced zoom of the image """

    def __init__(self, mainframe, **kw):
        """ Initialize the main Frame """
        ttk.Frame.__init__(self, master=mainframe)
        super().__init__(**kw)
        self.mask = None
        self.master.title('Segmentación de venas')
        self.master.protocol("WM_DELETE_WINDOW", self.onExit)
        # Variables

        self.image = None  # Imagen que se va a mostrar en formato PIL
        self.width = 0  # Ancho de la imagen
        self.height = 0  # Alto de la imagen

        self.polygon_points = np.array([])  # Puntos que forman el poligono
        self.isClosed = False  # Define si el poligono se cierra autmáticamente al poner los puntos
        self.thickness = 2  # Ancho de la línea
        self.is_enhanced = False  # Flag para saber si la imagen está mejorada
        self.is_skeletonized = False  # Flag para saber si la imagen está esqueletonizada
        self.is_subpixel = False
        self.initWelcomeUI()

    def initWelcomeUI(self):
        file = fd.askopenfilename()
        if file:
            self.filename = file
            self.opencv_image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            self.image = Image.open(self.filename)  # open image
            self.width, self.height = self.image.size
            self.initUiComponents()
            self.show_image()

    def initUiComponents(self):
        # Vertical and horizontal scrollbars for canvas
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')

        # Menu bar
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Abrir Archivo", command=self.openFileMenu)
        fileMenu.add_command(label="Salir", command=self.onExit)
        menubar.add_cascade(label="Archivo", menu=fileMenu)

        editMenu = Menu(menubar)
        editMenu.add_command(label="Auto mejorar", command=self.enhance)
        editMenu.add_command(label="Esqueletonizar", command=self.skeletonize)
        editMenu.add_command(label="Subpixel", command=self.subpixel)
        editMenu.add_command(label="Contraste y brillo", command=self.open_contrast_brightness_menu)
        menubar.add_cascade(label="Edicion", menu=editMenu)

        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scroll_y)  # bind scrollbars to the canvas
        hbar.configure(command=self.scroll_x)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.bindCanvasEvents()
        self.image = Image.open(self.filename)  # open image
        self.width, self.height = self.image.size
        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.show_image()

    def bindCanvasEvents(self):
        """ Bind events to the Canvas """
        self.canvas.bind('<Configure>', self.show_image)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.click_move)
        self.canvas.bind('<ButtonPress-3>', self.click_draw_polygon)
        self.canvas.bind('<B1-Motion>', self.move_to)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.canvas.bind('<Button-5>', self.wheel)  # only with Linux, wheel scroll down
        self.canvas.bind('<Button-4>', self.wheel)  # only with Linux, wheel scroll up
        self.master.bind('<KeyRelease-c>', self.clean)

    def scroll_y(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args, **kwargs)  # scroll vertically
        self.show_image()  # redraw the image

    def scroll_x(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args, **kwargs)  # scroll horizontally
        self.show_image()  # redraw the image

    def click_draw_polygon(self, event):
        print('Event::mouse2')
        print('Event click position is x={} y={}'.format(event.x, event.y))
        print('Real click position is x={} y={}'.format((event.x + self.x1) / self.imscale,
                                                        (event.y + self.y1) / self.imscale))
        print('Offset is x1={} y1={} x2={} y2={}'.format(self.x1, self.y1, self.x2, self.y2))
        # We only use positive real points
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            if self.is_enhanced or self.is_skeletonized:
                self.image = openCVToPIL(self.opencv_image)
                self.polygon_points = np.array([])
                self.is_enhanced = False
                self.is_skeletonized = False

            self.polygon_points = np.append(self.polygon_points,
                                            [(event.x + self.x1) / self.imscale, (event.y + self.y1) / self.imscale])

            print('Scale is {}'.format(self.imscale))
            pts = np.array(self.polygon_points).reshape((-1, 1, 2))

            # Creamos una linea para visualizar el area que se va a utilizar
            image_with_polygon = cv2.polylines(self.opencv_image.copy(), [pts.astype(np.int32)], isClosed=self.isClosed,
                                               color=color.red, thickness=self.thickness)
            # Creamos la máscara cerrando el poligono
            self.mask = cv2.fillPoly(np.zeros((self.height, self.width, 3)),
                                     [pts.astype(np.int32)], color=color.white)
            # Uncomment to test
            # plt.imshow(self.mask)
            # plt.show()
            self.image = openCVToPIL(image_with_polygon)  # open image
            self.width, self.height = self.image.size
            self.show_image()

    def clean(self, event):
        print('Event:clean')
        self.opencv_image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        self.image = openCVToPIL(self.opencv_image)  # open image
        self.polygon_points = np.array([])
        self.width, self.height = self.image.size
        self.show_image()

    def click_move(self, event):
        self.move_from(event)

    def move_from(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def move_to(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.show_image()  # redraw the image

    def wheel(self, event):
        """ Zoom with mouse wheel """
        print('Event::wheel')
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass  # Ok! Inside the image
        else:
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.num == 5 or event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30: return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        if event.num == 4 or event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale: return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.show_image()

    def show_image(self, event=None):
        """ Show image on the Canvas """
        bbox1 = self.canvas.bbox(self.container)  # get image area
        # Remove 1 pixel shift at the sides of the bbox1
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # get visible area of the canvas
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),  # get scroll region box
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # whole image in the visible area
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:  # whole image in the visible area
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # set scroll region
        x1 = max(bbox2[0] - bbox1[0], 0)  # get coordinates (x1,y1,x2,y2) of the image tile
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        self.x1 = bbox2[0] - bbox1[0]
        self.y1 = bbox2[1] - bbox1[1]
        self.x2 = x2
        self.y2 = y2

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:  # show image if it in the visible area
            x = min(int(x2 / self.imscale), self.width)  # sometimes it is larger on 1 pixel...
            y = min(int(y2 / self.imscale), self.height)  # ...and sometimes not
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imagetk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageid = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imagetk)
            self.canvas.lower(imageid)  # set image into background
            self.canvas.imagetk = imagetk  # keep an extra reference to prevent garbage-collection

    def onExit(self):
        if messagebox.askokcancel("Salir", "¿De seguro que quieres salir?"):
            self.master.destroy()
            sys.exit()

    def enhance(self):
        if len(self.polygon_points) > 1:
            self.is_skeletonized = mask.apply_enhance_to_roi(self.opencv_image, self.mask)
            pts = np.array(self.polygon_points).reshape((-1, 1, 2))
            image_with_polygon = cv2.polylines(self.is_skeletonized, [pts.astype(np.int32)], isClosed=self.isClosed,
                                               color=color.red, thickness=self.thickness)
            self.image = openCVToPIL(image_with_polygon)
            self.is_enhanced = True
            self.width, self.height = self.image.size
            self.show_image()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def skeletonize(self):
        if len(self.polygon_points) > 1:
            if self.is_enhanced:
                self.skeletonized = mask.apply_skeletonization_to_roi(self.is_skeletonized, self.mask, is_enhanced=True)
            else:
                self.skeletonized = mask.apply_skeletonization_to_roi(self.opencv_image, self.mask, is_enhanced=False)
            pts = np.array(self.polygon_points).reshape((-1, 1, 2))
            image_with_polygon = cv2.polylines(self.skeletonized, [pts.astype(np.int32)], isClosed=self.isClosed,
                                               color=color.red, thickness=self.thickness)

            self.image = openCVToPIL(image_with_polygon)
            self.is_skeletonized = True
            self.width, self.height = self.image.size
            self.show_image()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def subpixel(self):
        if len(self.polygon_points) > 1:
            self.is_skeletonized = mask.apply_subpixel_to_roi(self.opencv_image, self.mask)
            pts = np.array(self.polygon_points).reshape((-1, 1, 2))
            image_with_polygon = cv2.polylines(self.is_skeletonized, [pts.astype(np.int32)], isClosed=self.isClosed,
                                               color=color.red, thickness=self.thickness)
            self.image = openCVToPIL(image_with_polygon)
            self.is_subpixel = True
            self.width, self.height = self.image.size
            self.show_image()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def open_contrast_brightness_menu(self):
        d = BrightnessContrastDialog(self.master, self.opencv_image)
        self.master.wait_window(d.top)

    def openFileMenu(self):
        file = fd.askopenfilename()
        if file:
            self.filename = file
            self.opencv_image = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            self.image = Image.open(self.filename)  # open image
            self.width, self.height = self.image.size

            self.polygon_points = np.array([])  # Puntos que forman el poligono
            self.isClosed = False  # Define si el poligono se cierra autmáticamente al poner los puntos
            self.thickness = 2  # Ancho de la línea
            self.is_enhanced = False  # Flag para saber si la imagen está mejorada
            self.is_skeletonized = False  # Flag para saber si la imagen está esqueletonizada
            self.is_subpixel = False

            self.show_image()


def main():
    root = Tk()
    root.title('Segmentación de venas')
    root.geometry("1000x500")
    app = App(root, )
    app.mainloop()


if __name__ == '__main__':
    main()
