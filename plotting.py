import matplotlib.pyplot as plt


def plotArray(images):
    max_rows = len(images)
    figure = plt.figure(figsize=(1, 5), dpi=600 )
    figure.tight_layout()
    figure.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.75,
                           wspace=0.35)
    i = 1
    for image in images:
        figure.add_subplot(max_rows, 1, i)
        i = i + 1
        plt.imshow(image, cmap="gray")
        plt.axis("off")

    figure.show()
