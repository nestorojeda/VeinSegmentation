import sys
import tkinter as tk
from tkinter import Tk, messagebox, Menu
from tkinter import filedialog as fd

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk

import constants.colors as color
from Components.AutoScrollbar import AutoScrollbar
from Components.BrightnessContrastDialog import BrightnessContrastDialog
from Components.ReferencePointsDialog import ReferencePointsDialog
from Components.VeinMetricsModal import VeinMetricsModal
from Utils.Utils import openCVToPIL, PILtoOpenCV
from VeinSegmentation import Mask

drawing = False
ftypes = [('Imagen', '.png .jpeg .jpg')]
point_thickness = 8


# https://zetcode.com/tkinter/menustoolbars/
# https://solarianprogrammer.com/2018/04/20/python-opencv-show-image-tkinter-window/
# https://www.semicolonworld.com/question/55637/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width
# https://www.it-swarm-es.com/es/python/tkinter-canvas-zoom-move-pan/830432124/

class App(tk.Toplevel):
    """ Advanced zoom of the image """

    def __init__(self, mainframe, file=None, **kw):
        """ Initialize the main Frame """
        tk.Toplevel.__init__(self, master=mainframe)
        self.root = mainframe
        self.withdraw()
        self.mask = None
        self.master.title('Segmentación de venas')
        self.master.protocol("WM_DELETE_WINDOW", self.onExit)
        # Variables

        self.image = None  # Imagen que se va a mostrar en formato PIL
        self.zeroBrightnessAndContrastImage = None  # Imagen sin brillo ni contraste
        self.width = 0  # Ancho de la imagen
        self.height = 0  # Alto de la imagen

        self.brightnessValue = 0
        self.contrastValue = 0

        self.pixelSize = None
        self.rpd = None

        # ARRAYS DE PUNTOS
        self.polygonPoints = np.array([])  # Puntos que forman el poligono
        self.referencePoints = []  # Puntos de referencia
        self.measurePoints = []  # Puntos para medir

        self.isClosed = False  # Define si el poligono se cierra autmáticamente al poner los puntos
        self.thickness = 2  # Ancho de la línea

        self.isEnhanced = False  # Flag para saber si la imagen está mejorada
        self.blackPixels = None
        self.isSkeletonized = False  # Flag para saber si la imagen está esqueletonizada
        self.whitePixels = None
        self.isSubpixel = False

        self.filename = ''
        self.openCVImage = None
        self.originalOpenCVImage = None

        self.imscale = 1.0  # scale for the canvaas image
        self.delta = 1.3  # zoom magnitude

        # MODO PREDETERMINADO: DRAWING
        self.drawing = True
        self.measuring = tk.BooleanVar()
        self.measuring.set(False)
        self.selectReference = False
        if file:
            self.filename = file
            self.openCVImage = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
            self.openCVImage = cv2.cvtColor(self.openCVImage, cv2.COLOR_GRAY2RGB)
            self.originalOpenCVImage = self.openCVImage.copy()
            self.image = Image.open(self.filename)
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.width, self.height = self.image.size
            self.initUiComponents()
            self.showImage()
        else:
            self.initWelcomeUI()

    def onExit(self):
        if messagebox.askokcancel("Salir", "¿De seguro que quieres salir?"):
            self.master.destroy()
            sys.exit()

    def initWelcomeUI(self):
        self.master.withdraw()
        print("Starting Welcome UI")
        files = fd.askopenfilenames(filetypes=ftypes)
        if files:
            if len(files) == 1:
                self.master.deiconify()
                self.filename = files[0]
                self.openCVImage = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
                self.openCVImage = cv2.cvtColor(self.openCVImage, cv2.COLOR_GRAY2RGB)
                self.originalOpenCVImage = self.openCVImage.copy()
                self.image = Image.open(self.filename)
                self.zeroBrightnessAndContrastImage = self.image.copy()
                self.width, self.height = self.image.size
                self.initUiComponents()
                self.showImage()
            else:
                for file in files:
                    newInstance = tk.Toplevel()
                    newInstance.title('Segmentación de venas')
                    newInstance.geometry("1000x500")
                    app = App(newInstance, file)
                    self.master.withdraw()


        else:
            self.master.destroy()
            sys.exit()

    def initUiComponents(self):
        print("Starting UI")
        vbar = AutoScrollbar(self.master, orient='vertical')
        hbar = AutoScrollbar(self.master, orient='horizontal')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='we')

        # Menu bar
        menubar = Menu(self.master)
        self.master.config(menu=menubar)
        fileMenu = Menu(menubar, tearoff=0)
        fileMenu.add_command(label="Abrir", command=self.openFileMenu)
        fileMenu.add_command(label="Guardar", command=self.saveFileMenu)
        fileMenu.add_command(label="Salir", command=self.onExit)
        menubar.add_cascade(label="Archivo", menu=fileMenu)

        editMenu = Menu(menubar, tearoff=0)
        editMenu.add_command(label="Auto mejorar", command=self.enhance)
        editMenu.add_command(label="Esqueletonizar", command=self.skeletonize)
        editMenu.add_command(label="Subpixel", command=self.subpixel)
        editMenu.add_command(label="Contraste y brillo", command=self.openContrastBrightnessMenu)
        menubar.add_cascade(label="Edicion", menu=editMenu)

        measureMenu = Menu(menubar, tearoff=0)
        measureMenu.add_command(label="Seleccionar referencia", command=self.selectReferenceMode)
        measureMenu.add_checkbutton(label="Medir", variable=self.measuring,
                                    command=self.toggleMeasureMode)
        measureMenu.add_command(label="Información sobre la selección", command=self.selectionInfo)
        menubar.add_cascade(label="Medidas", menu=measureMenu)

        # Create canvas and put image on it
        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()  # wait till canvas is created
        vbar.configure(command=self.scrollY)  # bind scrollbars to the canvas
        hbar.configure(command=self.scrollX)
        # Make the canvas expandable
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.bindCanvasEvents()
        # Put image into container rectangle and use it to set proper coordinates to the image
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.showImage()

    ## EVENTOS ##
    def bindCanvasEvents(self):
        """ Bind events to the Canvas """
        self.canvas.bind('<Configure>', self.showImage)  # canvas is resized
        self.canvas.bind('<ButtonPress-1>', self.clickMove)
        self.canvas.bind('<ButtonPress-3>', self.clickRight)
        self.canvas.bind('<B1-Motion>', self.moveTo)
        self.canvas.bind('<MouseWheel>', self.wheel)  # with Windows and MacOS, but not Linux
        self.master.bind('<KeyRelease-c>', self.clean)

    def scrollY(self, *args, **kwargs):
        """ Scroll canvas vertically and redraw the image """
        self.canvas.yview(*args)  # scroll vertically
        self.showImage()  # redraw the image

    def scrollX(self, *args, **kwargs):
        """ Scroll canvas horizontally and redraw the image """
        self.canvas.xview(*args)  # scroll horizontally
        self.showImage()  # redraw the image

    def clickRight(self, event):
        if self.drawing:
            self.clickDrawPolygon(event)
        if self.selectReference:
            self.clickSelectReference(event)
        if self.measuring.get():
            self.clickSelectMeasure(event)

    def clickMove(self, event):
        self.moveFrom(event)

    def moveFrom(self, event):
        """ Remember previous coordinates for scrolling with the mouse """
        self.canvas.scan_mark(event.x, event.y)

    def moveTo(self, event):
        """ Drag (move) canvas to the new position """
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.showImage()  # redraw the image

    def showImage(self, event=None):
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
        print('bbox1: {}'.format(bbox1))
        print('bbox2: {}'.format(bbox2))
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
            x = min(int(x2 / self.imscale), self.width)
            y = min(int(y2 / self.imscale), self.height)
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imageTk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageId = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imageTk)
            self.canvas.lower(imageId)  # set image into background
            self.canvas.imagetk = imageTk  # keep an extra reference to prevent garbage-collection

    def wheel(self, event):
        """ Zoom with mouse wheel """
        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # get image area
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass  # Ok! Inside the image
        else:
            return  # zoom only inside image area
        scale = 1.0
        # Respond to Linux (event.num) or Windows (event.delta) wheel event
        if event.delta == -120:  # scroll down
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30:
                return  # image is less than 30 pixels
            self.imscale /= self.delta
            scale /= self.delta
        if event.delta == 120:  # scroll up
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale:
                return  # 1 pixel is bigger than the visible area
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # rescale all canvas objects
        self.showImage()

    def clickSelectReference(self, event):
        if self.rpd:
            self.rpd.cancel()
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            clickX = int((event.x + self.x1) / self.imscale)
            clickY = int((event.y + self.y1) / self.imscale)
            self.referencePoints.append((clickX, clickY))

            if len(self.referencePoints) <= 1:
                self.imageWithPoints = cv2.circle(self.openCVImage.copy(),
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)
                self.image = openCVToPIL(self.imageWithPoints)  # open image
                self.showImage()

            if len(self.referencePoints) == 2:
                self.imageWithPoints = cv2.circle(self.imageWithPoints,
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)

                image_with_line = cv2.line(self.imageWithPoints,
                                           self.referencePoints[0], self.referencePoints[1],
                                           color=color.red,
                                           thickness=self.thickness)

                self.image = openCVToPIL(image_with_line)  # open image
                self.showImage()
                self.rpd = ReferencePointsDialog(self.master)
                self.master.wait_window(self.rpd.top)
                self.referencePoints = []
                if self.pixelSize:
                    self.toggleMeasureMode()
                else:
                    self.selectDrawingMode()

    def clickSelectMeasure(self, event):
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            clickX = int((event.x + self.x1) / self.imscale)
            clickY = int((event.y + self.y1) / self.imscale)
            self.measurePoints.append((clickX, clickY))

            if len(self.measurePoints) <= 1:
                self.imageWithPoints = cv2.circle(self.openCVImage.copy(),
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)
                self.image = openCVToPIL(self.imageWithPoints)  # open image
                self.showImage()

            if len(self.measurePoints) == 2:
                self.imageWithPoints = cv2.circle(self.imageWithPoints,
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)
                imageWithLine = cv2.line(self.imageWithPoints,
                                         self.measurePoints[0], self.measurePoints[1],
                                         color=color.red,
                                         thickness=self.thickness)

                pixel_distance = np.math.sqrt(
                    (self.measurePoints[1][0] - self.measurePoints[0][0]) ** 2 +
                    (self.measurePoints[1][1] - self.measurePoints[0][1]) ** 2)
                print("Pixel distance betweeen points is: {} pixels".format(pixel_distance))
                distance = pixel_distance * self.pixelSize
                print("Real istance betweeen points is: {} cm".format(distance))

                self.image = openCVToPIL(imageWithLine)  # open image
                self.showImage()
                self.measurePoints = []
                messagebox.showinfo(message="La distancia entre los dos puntos es de {} cm".format(distance),
                                    title="Distancia")

    def clickDrawPolygon(self, event):
        print('Event click position is x={} y={}'.format(event.x, event.y))
        print('Real click position is x={} y={}'.format((event.x + self.x1) / self.imscale,
                                                        (event.y + self.y1) / self.imscale))
        print('Offset is x1={} y1={} x2={} y2={}'.format(self.x1, self.y1, self.x2, self.y2))
        print('Scale is {}'.format(self.imscale))
        # We only use positive real points
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            if self.isEnhanced or self.isSkeletonized or self.isSubpixel:
                self.clean()
                self.polygonPoints = np.array([])
                self.isEnhanced = False
                self.isSkeletonized = False

            self.polygonPoints = np.append(self.polygonPoints,
                                           [(event.x + self.x1) / self.imscale, (event.y + self.y1) / self.imscale])

            pts = np.array(self.polygonPoints).reshape((-1, 1, 2))

            # Creamos una linea para visualizar el area que se va a utilizar
            imageWithPolygon = cv2.polylines(self.openCVImage.copy(),
                                             [pts.astype(np.int32)], isClosed=self.isClosed,
                                             color=color.red, thickness=self.thickness)
            # Creamos la máscara cerrando el poligono
            self.mask = cv2.fillPoly(np.zeros((self.height, self.width, 3)),
                                     [pts.astype(np.int32)], color=color.white)

            self.image = openCVToPIL(imageWithPolygon)  # open image
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.showImage()

    ## CAMBIOS DE MODO ##
    def selectReferenceMode(self):
        print('Mode changed to reference mode')
        if self.pixelSize:
            messageBox = tk.messagebox.askquestion('Aviso', '¿Estás seguro que que deseas rehacer la '
                                                            'referencia?',
                                                   icon='warning')
            if messageBox == 'yes':
                self.pixelSize = None
                self.drawing = False
                self.measuring.set(False)
                self.selectReference = True

        else:
            self.drawing = False
            self.measuring.set(False)
            self.selectReference = True

    def toggleMeasureMode(self):
        if self.drawing or self.selectReference:
            if self.pixelSize:
                print('Mode changed to measure mode')
                self.drawing = False
                self.measuring.set(True)
                self.selectReference = False
            else:
                messagebox.showinfo('Aviso',
                                    'Debes seleccionar antes una referencia', icon='error')
                self.measuring.set(False)
        else:
            self.selectDrawingMode()

    def selectDrawingMode(self):
        print('Mode changed to drawing mode')
        self.drawing = True
        self.measuring.set(False)
        self.selectReference = False
        self.clean()

    def clean(self, event=None):
        print('Clean')
        self.whitePixels = None
        self.blackPixels = None
        self.openCVImage = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        self.openCVImage = cv2.cvtColor(self.openCVImage, cv2.COLOR_GRAY2RGB)
        self.originalOpenCVImage = self.openCVImage.copy()
        self.image = openCVToPIL(self.openCVImage)  # open image
        self.zeroBrightnessAndContrastImage = self.image.copy()
        self.polygonPoints = np.array([])
        self.showImage()

    ## PROCESAMIENTOS ##
    def enhance(self):
        if len(self.polygonPoints) > 1:
            enhanced, self.blackPixels = Mask.applyEnhanceToROI(self.originalOpenCVImage.copy(), self.mask)
            pts = np.array(self.polygonPoints).reshape((-1, 1, 2))
            image_with_polygon = cv2.polylines(enhanced, [pts.astype(np.int32)], isClosed=self.isClosed,
                                               color=color.red, thickness=self.thickness)
            self.image = openCVToPIL(image_with_polygon)
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.openCVImage = enhanced
            self.isEnhanced = True
            self.showImage()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def skeletonize(self):
        if len(self.polygonPoints) > 1:
            self.skeletonized, self.whitePixels = Mask.applySkeletonizationToROI(self.originalOpenCVImage.copy(),
                                                                                 self.mask)
            pts = np.array(self.polygonPoints).reshape((-1, 1, 2))
            imageWithPolygon = cv2.polylines(self.skeletonized, [pts.astype(np.int32)], isClosed=self.isClosed,
                                             color=color.red, thickness=self.thickness)

            self.image = openCVToPIL(imageWithPolygon)
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.openCVImage = self.skeletonized
            self.isSkeletonized = True
            self.showImage()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def subpixel(self):
        if len(self.polygonPoints) > 1:
            self.subpixelImage = Mask.applySubpixelToROI((self.originalOpenCVImage.astype(float)).copy(),
                                                         self.mask)
            pts = np.array(self.polygonPoints).reshape((-1, 1, 2))
            imageWithPolygon = cv2.polylines(self.subpixelImage, [pts.astype(np.int32)], isClosed=self.isClosed,
                                             color=color.red, thickness=self.thickness)

            self.image = openCVToPIL(imageWithPolygon)
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.openCVImage = self.subpixelImage
            self.isSubpixel = True
            self.showImage()
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def openContrastBrightnessMenu(self):
        d = BrightnessContrastDialog(self.master, self.zeroBrightnessAndContrastImage.copy())
        self.master.wait_window(d.top)

    def selectionInfo(self):
        if len(self.polygonPoints) > 1:
            metrics = VeinMetricsModal(self.master)
            self.master.wait_window(metrics.top)
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def openFileMenu(self):
        file = fd.askopenfilename(filetypes=ftypes)
        if file:
            messageBox = tk.messagebox.askquestion('Aviso', '¿Deseas abrir la imagen en una nueva ventana?',
                                               icon='warning')
            if messageBox == 'yes':
                newInstance = tk.Toplevel()
                newInstance.title('Segmentación de venas')
                newInstance.geometry("1000x500")
                app = App(newInstance, file)

            else:
                self.filename = file
                self.clean()
                self.showImage()

    def saveFileMenu(self):
        filename = fd.asksaveasfilename(filetypes=ftypes,
                                        defaultextension='.png')
        if filename:
            cv2.imwrite(filename, PILtoOpenCV(self.image))


def main():
    root = Tk()
    root.title('Segmentación de venas')
    root.geometry("1000x500")
    app = App(root, )
    app.mainloop()


if __name__ == '__main__':
    main()