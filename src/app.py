import sys
import tkinter as tk
from tkinter import Tk, messagebox, Menu
from tkinter import filedialog as fd

import cv2
import numpy as np
from PIL import Image
from PIL import ImageTk

import constants.colors as color
from src.Components.AutoScrollbar import AutoScrollbar
from src.Components.Modals.BrightnessContrastModal import BrightnessContrastDialog
from src.Components.Modals.ReferencePointsModal import ReferencePointsDialog
from src.Components.Modals.SkeletonizationControlModal import SkeletonizationControl
from src.Components.Modals.VeinMetricsModal import VeinMetricsModal
from src.Utils.Utils import openCVToPIL, PILtoOpenCV
from src.VeinSegmentation.Processing import Processing

drawing = False
ftypes = [('Imagen', '.png .jpeg .jpg')]
point_thickness = 8
windowsOpen = 0  # Indica el número de ventanas que hay abiertas


# https://zetcode.com/tkinter/menustoolbars/
# https://solarianprogrammer.com/2018/04/20/python-opencv-show-image-tkinter-window/
# https://www.semicolonworld.com/question/55637/how-to-get-tkinter-canvas-to-dynamically-resize-to-window-width
# https://www.it-swarm-es.com/es/python/tkinter-canvas-zoom-move-pan/830432124/

class App(tk.Toplevel):
    global windowsOpen

    def __init__(self, mainframe, file=None, **kw):
        """ Initialize the main Frame """
        tk.Toplevel.__init__(self, master=mainframe)
        self.root = mainframe
        self.withdraw()
        self.mask = None
        self.master.title('Segmentación de venas')
        self.master.protocol("WM_DELETE_WINDOW", self.onExit)

        # Inicialización de variables
        self.image = None  # Imagen que se va a mostrar en formato PIL
        self.zeroBrightnessAndContrastImage = None  # Imagen sin brillo ni contraste
        self.width = 0  # Ancho de la imagen
        self.height = 0  # Alto de la imagen

        self.brightnessValue = 0
        self.contrastValue = 0

        self.pixelSize = None  # referencia de la medida
        self.referencePointsDialog = None  # dialogo de los puntos de referencia

        # ARRAYS DE PUNTOS
        self.polygonPoints = np.array([])  # Puntos que forman el poligono
        self.referencePoints = []  # Puntos de referencia
        self.measurePoints = []  # Puntos para medir
        self.correctionPoints = []

        self.isClosed = False  # Define si el poligono se cierra autmáticamente al poner los puntos
        self.thickness = 2  # Ancho de la línea

        self.isEnhanced = False  # Flag para saber si la imagen está mejorada
        self.isSkeletonized = False  # Flag para saber si la imagen está esqueletonizada
        self.isSubpixel = False  # Flag para saber si a la imagen se le ha aplicado el algoritmo de subpixel

        self.imscale = 1.0  # Escala del canvas
        self.delta = 1.3  # Magnitud del zoom
        self.lineWidth = 1  # Ancho de la linea de los contornos
        self.drawing = True  # Flag que indica si se esta seleccionando poligono
        self.measuring = tk.BooleanVar()  # Flag que indica si se esta midiendo
        self.measuring.set(False)
        self.selectReference = False  # Flag que indica si se está seleccionando la medida
        self.metrics = None
        self.skelControl = None

        self.correctSkeletonization = tk.BooleanVar()  # Flag que indica si se esta midiendo
        self.correctSkeletonization.set(False)

        self.imageWithPoints = None  # Imagen con los puntos dibujados

        self.processing = None  # Instancia de la clase Processig
        self.preMeasureImage = None
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
        """ Manejo del evento de salir de la página """
        global windowsOpen
        if windowsOpen - 1 == 0:
            if messagebox.askokcancel("Salir", "¿De seguro que quieres salir?"):
                windowsOpen -= 1
                self.master.destroy()
                sys.exit()
        else:
            windowsOpen -= 1
            self.master.withdraw()

    def initWelcomeUI(self):
        """ Inicialización de la vista de selección de ficheros """
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
                    App(newInstance, file)
                    self.master.withdraw()
        else:
            self.master.destroy()
            sys.exit()

    def initUiComponents(self):
        """ Inicialización de los componentes de la interfaz """
        global windowsOpen
        windowsOpen += 1

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
        editMenu.add_checkbutton(label="Corregir esqueletonización", variable=self.correctSkeletonization,
                                 command=self.toggleCorrectSkeletonization)
        editMenu.add_command(label="Subpixel", command=self.subpixel)
        editMenu.add_command(label="Contraste y brillo", command=self.openContrastBrightnessMenu)
        menubar.add_cascade(label="Edicion", menu=editMenu)

        measureMenu = Menu(menubar, tearoff=0)
        measureMenu.add_command(label="Seleccionar referencia", command=self.selectReferenceMode)
        measureMenu.add_checkbutton(label="Medir", variable=self.measuring,
                                    command=self.toggleMeasureMode)
        measureMenu.add_command(label="Información sobre la selección", command=self.selectionInfo)
        menubar.add_cascade(label="Medidas", menu=measureMenu)

        self.canvas = tk.Canvas(self.master, highlightthickness=0,
                                xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        self.canvas.grid(row=0, column=0, sticky='nswe')
        self.canvas.update()
        vbar.configure(command=self.scrollY)
        hbar.configure(command=self.scrollX)
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)
        self.bindCanvasEvents()
        self.container = self.canvas.create_rectangle(0, 0, self.width, self.height, width=0)
        self.showImage()

    ## EVENTOS ##
    def bindCanvasEvents(self):
        """ Eventos del Canvas """

        self.canvas.bind('<Configure>', self.showImage)  # Cambio de tamaño del canvas
        self.canvas.bind('<ButtonPress-1>', self.moveFrom)  # Evento del click antes de mover la image
        self.canvas.bind('<ButtonPress-3>', self.clickRight)  # Evento del click derecho
        self.canvas.bind('<B1-Motion>', self.moveTo)  # Movimiento del raton para mover la imagen
        self.canvas.bind('<MouseWheel>', self.wheel)  # Evento de la rueda del ratón
        self.master.bind('<KeyRelease-c>', self.clean)  # Evento de limpiar la imagen

    def scrollY(self, *args, **kwargs):
        """ Scroll vertical """

        self.canvas.yview(*args)
        self.showImage()

    def scrollX(self, *args, **kwargs):
        """ Scroll horizontal """

        self.canvas.xview(*args)
        self.showImage()

    def clickRight(self, event):
        """ Manejo del evento del click derecho """
        if self.drawing:
            self.clickDrawPolygon(event)
            return
        if self.selectReference:
            self.clickSelectReference(event)
            return
        if self.measuring.get():
            self.clickSelectMeasure(event)
            return
        if self.correctSkeletonization.get():
            self.clickCorrectSkeletonization(event)
            return

    def moveFrom(self, event):
        """ Guardamos la coordenada desde la que se ha lanzado el evento """

        self.canvas.scan_mark(event.x, event.y)

    def moveTo(self, event):
        """ Guardamos el lugar al que movemos el ratón """

        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.showImage()

    def showImage(self, event=None):
        """ Muestra la imagen en el canvas """

        bbox1 = self.canvas.bbox(self.container)
        # Quitamos un pixel de desplazamiento
        bbox1 = (bbox1[0] + 1, bbox1[1] + 1, bbox1[2] - 1, bbox1[3] - 1)
        bbox2 = (self.canvas.canvasx(0),  # Area visible
                 self.canvas.canvasy(0),
                 self.canvas.canvasx(self.canvas.winfo_width()),
                 self.canvas.canvasy(self.canvas.winfo_height()))
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]
        if bbox[0] == bbox2[0] and bbox[2] == bbox2[2]:  # Toda la imagen está en el area visible
            bbox[0] = bbox1[0]
            bbox[2] = bbox1[2]
        if bbox[1] == bbox2[1] and bbox[3] == bbox2[3]:
            bbox[1] = bbox1[1]
            bbox[3] = bbox1[3]
        self.canvas.configure(scrollregion=bbox)  # Region del scroll
        x1 = max(bbox2[0] - bbox1[0], 0)  # Coordenadas del viewport
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        self.x1 = bbox2[0] - bbox1[0]
        self.y1 = bbox2[1] - bbox1[1]
        self.x2 = x2
        self.y2 = y2

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:
            x = min(int(x2 / self.imscale), self.width)
            y = min(int(y2 / self.imscale), self.height)
            image = self.image.crop((int(x1 / self.imscale), int(y1 / self.imscale), x, y))
            imageTk = ImageTk.PhotoImage(image.resize((int(x2 - x1), int(y2 - y1))))
            imageId = self.canvas.create_image(max(bbox2[0], bbox1[0]), max(bbox2[1], bbox1[1]),
                                               anchor='nw', image=imageTk)
            self.canvas.lower(imageId)  # Ponemos la imagen de fondo
            self.canvas.imagetk = imageTk  # Guardamos una referencia extra para el garbage collector

    def wheel(self, event):
        """ Zoom con la rueda del ratón """

        x = self.canvas.canvasx(event.x)
        y = self.canvas.canvasy(event.y)
        bbox = self.canvas.bbox(self.container)  # Cogemos el area de la imagen
        if bbox[0] < x < bbox[2] and bbox[1] < y < bbox[3]:
            pass  # Ok! Inside the image
        else:
            return  # zoom only inside image area
        scale = 1.0
        if event.delta == -120:  # Scroll hacia abajo
            i = min(self.width, self.height)
            if int(i * self.imscale) < 30:
                return  # la imagen es menor a 30 pixeles
            self.imscale /= self.delta
            scale /= self.delta
        if event.delta == 120:  # Scroll hacia arrina
            i = min(self.canvas.winfo_width(), self.canvas.winfo_height())
            if i < self.imscale:
                return  # 1 pixel es mayor al area visible
            self.imscale *= self.delta
            scale *= self.delta
        self.canvas.scale('all', x, y, scale, scale)  # Reescalamos el canvas
        if self.isSkeletonized and self.skelControl:
            if self.skelControl.applyContour.get():
                if self.imscale < 0.5:
                    newLineWidth = 3
                elif 0.5 > self.imscale > 1:
                    newLineWidth = 2
                else:
                    newLineWidth = 1
                if newLineWidth != self.lineWidth:
                    self.lineWidth = newLineWidth
                    self.drawLines(self.processing.skeletonSettings(self.skelControl.applyCenterline.get(),
                                                                    self.skelControl.applyContour.get(),
                                                                    self.skelControl.applyTrasparency.get(),
                                                                    self.lineWidth))
        self.showImage()

    def clickSelectReference(self, event):
        """ Manejo del evento de selección de la referencia """

        if self.referencePointsDialog:
            self.referencePointsDialog.cancel()
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            clickX = int((event.x + self.x1) / self.imscale)
            clickY = int((event.y + self.y1) / self.imscale)
            self.referencePoints.append((clickX, clickY))

            if len(self.referencePoints) <= 1:
                self.imageWithPoints = cv2.circle(self.openCVImage.copy(),
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)
                self.image = openCVToPIL(self.imageWithPoints)
                self.showImage()

            if len(self.referencePoints) == 2:
                self.imageWithPoints = cv2.circle(self.imageWithPoints,
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)

                image_with_line = cv2.line(self.imageWithPoints,
                                           self.referencePoints[0], self.referencePoints[1],
                                           color=color.red,
                                           thickness=self.thickness)

                self.image = openCVToPIL(image_with_line)
                self.showImage()
                self.referencePointsDialog = ReferencePointsDialog(self.master)
                self.master.wait_window(self.referencePointsDialog.top)
                self.referencePoints = []
                if self.pixelSize:
                    self.toggleMeasureMode()
                else:
                    self.selectDrawingMode()

    def clickSelectMeasure(self, event):
        """ Manejo del evento de selección de medida """

        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            clickX = int((event.x + self.x1) / self.imscale)
            clickY = int((event.y + self.y1) / self.imscale)
            self.measurePoints.append((clickX, clickY))

            if len(self.measurePoints) <= 1:
                self.imageWithPoints = cv2.circle(self.openCVImage.copy(),
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=point_thickness)
                self.image = openCVToPIL(self.imageWithPoints)
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
                distance = pixel_distance * self.pixelSize

                self.image = openCVToPIL(imageWithLine)
                self.showImage()
                self.measurePoints = []
                messagebox.showinfo(message="La distancia entre los dos puntos es de {} cm".format(distance),
                                    title="Distancia")

    def clickDrawPolygon(self, event):
        """ Manejo del evento de dibujar el poligono para la mascara """

        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            if self.isEnhanced or self.isSkeletonized or self.isSubpixel:
                self.skelControl.top.destroy()
                self.clean()

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

            self.processing = Processing(self.originalOpenCVImage, self.mask)

            self.image = openCVToPIL(imageWithPolygon)
            self.zeroBrightnessAndContrastImage = self.image.copy()
            self.showImage()

    def clickCorrectSkeletonization(self, event):
        if (event.x + self.x1) / self.imscale >= 0 and (event.y + self.y1) / self.imscale >= 0:
            clickX = int((event.x + self.x1) / self.imscale)
            clickY = int((event.y + self.y1) / self.imscale)

            self.correctionPoints.append((clickX, clickY))

            if len(self.correctionPoints) <= 1:
                self.imageWithPoints = cv2.circle(self.openCVImage.copy(),
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=1)
                self.image = openCVToPIL(self.imageWithPoints)
                self.showImage()

            if len(self.correctionPoints) == 2:
                self.imageWithPoints = cv2.circle(self.imageWithPoints,
                                                  (clickX, clickY), radius=0, color=(0, 0, 255),
                                                  thickness=1)
                imageWithLine = cv2.line(self.imageWithPoints,
                                         self.correctionPoints[0], self.correctionPoints[1],
                                         color=(0, 0, 255),
                                         thickness=1)

                self.processing.correctSkeleton(self.correctionPoints)

                self.image = openCVToPIL(imageWithLine)
                self.showImage()
                self.correctionPoints = []
                self.toggleCorrectSkeletonization()

    def selectReferenceMode(self):
        """ Cambio de modo a selección de referencia """

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
                self.preMeasureImage = self.image.copy()

        else:
            self.drawing = False
            self.measuring.set(False)
            self.selectReference = True
            self.preMeasureImage = self.image.copy()

    def toggleMeasureMode(self):
        """ Cambio de modo a medida """

        if self.drawing or self.selectReference:
            if self.drawing:
                self.preMeasureImage = self.image.copy()
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

    def toggleCorrectSkeletonization(self):
        if self.isSkeletonized and self.drawing:
            self.correctSkeletonization.set(True)
            self.drawing = False
        elif self.correctSkeletonization.get():
            self.selectDrawingMode()
            return
        else:
            messagebox.showinfo('Aviso', 'Debes haber aplicado la esqueletonización', icon='error')

    def selectDrawingMode(self):
        """ Cambio de modo a dibujar """
        print('Mode changed to drawing mode')
        self.drawing = True
        self.measuring.set(False)
        self.correctSkeletonization.set(False)
        self.selectReference = False
        if self.preMeasureImage is not None:
            self.image = self.preMeasureImage.copy()
            self.openCVImage = PILtoOpenCV(self.preMeasureImage.copy())
            self.showImage()
            self.preMeasureImage =None

    def clean(self, event=None):
        """ Limpieza del canvas y los procesamientos """

        print('Clean')
        if self.skelControl:
            self.skelControl.top.destroy()
        if self.metrics:
            self.metrics.top.destroy()

        self.isEnhanced = False
        self.isSkeletonized = False
        self.isSubpixel = False
        self.correctSkeletonization.set(False)
        self.openCVImage = cv2.imread(self.filename, cv2.IMREAD_GRAYSCALE)
        self.openCVImage = cv2.cvtColor(self.openCVImage, cv2.COLOR_GRAY2RGB)
        self.originalOpenCVImage = self.openCVImage.copy()
        self.image = openCVToPIL(self.openCVImage)
        self.zeroBrightnessAndContrastImage = self.image.copy()
        self.polygonPoints = np.array([])
        self.showImage()

    def enhance(self):
        """ Mejora automática de la imagen """

        if len(self.polygonPoints) > 1:
            if self.skelControl:
                self.skelControl.top.destroy()

            enhanced = self.processing.enhance()
            self.drawLines(enhanced)
            self.isEnhanced = True
            self.showImage()
            self.brightnessValue = 0
            self.contrastValue = 0
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def skeletonize(self):
        """ Esqueletonización de la imagen """

        if len(self.polygonPoints) > 1:
            skeletonized = self.processing.skeletonization()
            self.drawLines(skeletonized)
            self.isSkeletonized = True
            self.showImage()
            self.brightnessValue = 0
            self.contrastValue = 0
            self.skelControl = SkeletonizationControl(self.master)
            self.master.wait_window(self.skelControl.top)
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def subpixel(self):
        """ Detección de bordes por subpixel de la imagen """

        if len(self.polygonPoints) > 1:
            if self.skelControl:
                self.skelControl.top.destroy()

            try:
                subpixelImage = self.processing.subpixel()
            except UnboundLocalError:
                messagebox.showerror("Error", "Ha ocurrido un error mientras se realizaba el procesamiento")
                self.clean()
                return

            self.drawLines(subpixelImage)
            self.isSubpixel = True
            self.showImage()
            self.brightnessValue = 0
            self.contrastValue = 0
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def drawLines(self, image):
        """ Dibujo del poligono en la imagen"""

        pts = np.array(self.polygonPoints).reshape((-1, 1, 2))
        imageWithPolygon = cv2.polylines(image, [pts.astype(np.int32)], isClosed=self.isClosed,
                                         color=color.red, thickness=self.thickness)
        self.image = openCVToPIL(imageWithPolygon)
        self.zeroBrightnessAndContrastImage = self.image.copy()
        self.openCVImage = image

    def openContrastBrightnessMenu(self):
        """ Lanzamiento de la ventana de brillo y contraste """
        d = BrightnessContrastDialog(self.master, self.zeroBrightnessAndContrastImage.copy())
        self.master.wait_window(d.top)

    def selectionInfo(self):
        """ Lanzamiento de la ventana de información sobre la seleccion """
        if len(self.polygonPoints) > 1:
            self.metrics = VeinMetricsModal(self.master)
            self.master.wait_window(self.metrics.top)
        else:
            messagebox.showerror("Error", "Debes seleleccionar un polígono")

    def openFileMenu(self):
        """ Lanzamiento de la ventana de selección de ficheros """
        files = fd.askopenfilenames(filetypes=ftypes)
        if files:
            for file in files:
                newInstance = tk.Toplevel()
                newInstance.title('Segmentación de venas')
                newInstance.geometry("1000x500")
                App(newInstance, file)

    def saveFileMenu(self):
        """ Lanzamiento de la ventana de guadar una imagen """
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
