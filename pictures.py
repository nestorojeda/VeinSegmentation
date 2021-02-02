import matplotlib.pyplot as plt
import numpy as np


class Picture:
    def __init__(self, image, title):
        self.image = image
        self.title = title


def get_img_from_fig(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def show(pictures):
    max_rows = round((len(pictures) / 2))
    figure = plt.figure()
    figure.tight_layout()
    figure.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                           wspace=0.35)
    i = 1
    for picture in pictures:
        f1 = figure.add_subplot(max_rows, 2, i)
        i = i + 1
        plt.imshow(picture.image, cmap="gray")
        plt.axis('off')
        f1.set_title(picture.title)

    figure.show()
