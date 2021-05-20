import matplotlib.pyplot as plt


def plotArray(images, title=""):
    max_rows = len(images)
    figure = plt.figure(figsize=(1, max_rows-1), dpi=600)
    if title != "":
        figure.suptitle(title, fontsize=5)
    figure.tight_layout()
    figure.subplots_adjust(top=0.95, left=0.10, right=0.95)
    i = 1
    for image in images:
        figure.add_subplot(max_rows, 1, i)
        i = i + 1
        plt.imshow(image, cmap="gray")
        plt.xticks(fontsize=2)
        plt.yticks(fontsize=2)

    # plt.axis("off")

    figure.show()
