from tkinter import Checkbutton, Toplevel, BooleanVar


class SkeletonizationControl:
    def __init__(self, parent):
        self.parent = parent.children['!app']
        self.top = Toplevel(parent)
        self.top.transient(parent)
        title = "Ajustes"
        self.top.title(title)
        self.top.protocol("WM_DELETE_WINDOW", self.disableExit)

        self.applyTrasparency = BooleanVar(self.top)
        self.applyContour = BooleanVar(self.top)

        checkboxContour = Checkbutton(self.top, text="Contornos",
                                      variable=self.applyContour,
                                      command=self.checkBoxClicked).pack()
        checkboxTransparency = Checkbutton(self.top, text="Transparencia",
                                           variable=self.applyTrasparency,
                                           command=self.checkBoxClicked).pack()

    def cancel(self, event=None):
        self.parent.focus_set()
        self.top.destroy()

    def checkBoxClicked(self):
        self.parent.drawLines(self.parent.processing.skeletonSettings(self.applyContour.get(), self.applyTrasparency.get()))
        self.parent.isSkeletonized = True
        self.parent.showImage()

    def disableExit(self):
        pass
