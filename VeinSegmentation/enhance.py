import copy
import math
import warnings

import cv2
import numpy as np
import scipy.ndimage.filters as flt

white = 255.
black = 0.


def gaborFiltering(img):
    img = img.astype(np.float32) / 255
    kernel = cv2.getGaborKernel((21, 21), 5, 1, 10, 1, 0, cv2.CV_32F)
    kernel /= math.sqrt((kernel * kernel).sum())
    filtered = cv2.filter2D(img, -1, kernel)
    return filtered


def segmentation(img, n_clusters=2):
    Z = img.reshape((-1, 1))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n_clusters
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    return res2


def skeletonization(img, niter=100):
    enhanced_segm = segmentation(img, n_clusters=2)

    ret, img = cv2.threshold(enhanced_segm.astype(np.uint8), 127, 255, 0)
    img = cv2.bitwise_not(img)

    img = img.astype(np.uint8)
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    # Get a Cross Shaped Kernel
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    i = 0
    while i < niter:
        i = i + 1
        # Step 1: Substract open from the original image
        open = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, open)
        # Step 2: Erode the original image and refine the skeleton
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
        # Step 3: If there are no white pixels left ie.. the image has been completely eroded, quit the loop
        if cv2.countNonZero(img) == 0:
            break
    return skel


def enhance_medical_image(image, clip_limit=5, tile_grid_size=5, use_clahe=True):
    """
    # Based in https://ieeexplore.ieee.org/document/6246971

    Enhanced medical image

    Usage:
    imgout = enhance_medical_image(image, clip_limit, tile_grid_size)

    Arguments:
            image    - input image
            clip_limit  -
            tile_grid_size  - size of the grid for the CLAHE

    Returns:
            imgout   - ehanced image

    """
    image = cv2.fastNlMeansDenoising(image.astype(np.uint8))
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    image = anisodiff(image, niter=1, kappa=50, gamma=0.25, step=(1., 1.), sigma=0, option=1)
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
        image = clahe.apply(image.astype(np.uint8))
        del clahe
    else:
        image = cv2.equalizeHist(image.astype(np.uint8))
    return image.astype(float)


def anisodiff(img, niter=1, kappa=50, gamma=0.1, step=(1., 1.), sigma=0, option=1, ploton=False):
    """
    Anisotropic diffusion.

    Usage:
    imgout = anisodiff(im, niter, kappa, gamma, option)

    Arguments:
            img    - input image
            niter  - number of iterations
            kappa  - conduction coefficient 20-100 ?
            gamma  - max value of .25 for stability
            step   - tuple, the distance between adjacent pixels in (y,x)
            option - 1 Perona Malik diffusion equation No 1
                     2 Perona Malik diffusion equation No 2
            ploton - if True, the image will be plotted on every iteration

    Returns:
            imgout   - diffused image.

    kappa controls conduction as a function of gradient.  If kappa is low
    small intensity gradients are able to block conduction and hence diffusion
    across step edges.  A large value reduces the influence of intensity
    gradients on conduction.

    gamma controls speed of diffusion (you usually want it at a maximum of
    0.25)

    step is used to scale the gradients in case the spacing between adjacent
    pixels differs in the x and y axes

    Diffusion equation 1 favours high contrast edges over low contrast ones.
    Diffusion equation 2 favours wide regions over smaller ones.

    Reference:
    P. Perona and J. Malik.
    Scale-space and edge detection using ansotropic diffusion.
    IEEE Transactions on Pattern Analysis and Machine Intelligence,
    12(7):629-639, July 1990.

    Original MATLAB code by Peter Kovesi
    School of Computer Science & Software Engineering
    The University of Western Australia
    pk @ csse uwa edu au
    <http://www.csse.uwa.edu.au>

    Translated to Python and optimised by Alistair Muldal
    Department of Pharmacology
    University of Oxford
    <alistair.muldal@pharm.ox.ac.uk>

    June 2000  original version.
    March 2002 corrected diffusion eqn No 2.
    July 2012 translated to Python
    """

    # ...you could always diffuse each color channel independently if you
    # really want
    if img.ndim == 3:
        warnings.warn("Only grayscale images allowed, converting to 2D matrix")
        img = img.mean(2)

    # initialize output array
    img = img.astype('float32')
    imgout = img.copy()

    # initialize some internal variables
    deltaS = np.zeros_like(imgout)
    deltaE = deltaS.copy()
    NS = deltaS.copy()
    EW = deltaS.copy()
    gS = np.ones_like(imgout)
    gE = gS.copy()

    # create the plot figure, if requested
    if ploton:
        import pylab as pl

        fig = pl.figure(figsize=(20, 5.5), num="Anisotropic diffusion")
        ax1, ax2 = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)

        ax1.imshow(img, interpolation='nearest')
        ih = ax2.imshow(imgout, interpolation='nearest', animated=True)
        ax1.set_title("Original image")
        ax2.set_title("Iteration 0")

        fig.canvas.draw()

    for ii in np.arange(1, niter):

        # calculate the diffs
        deltaS[:-1, :] = np.diff(imgout, axis=0)
        deltaE[:, :-1] = np.diff(imgout, axis=1)

        if 0 < sigma:
            deltaSf = flt.gaussian_filter(deltaS, sigma);
            deltaEf = flt.gaussian_filter(deltaE, sigma);
        else:
            deltaSf = deltaS;
            deltaEf = deltaE;

        # conduction gradients (only need to compute one per dim!)
        if option == 1:
            gS = np.exp(-(deltaSf / kappa) ** 2.) / step[0]
            gE = np.exp(-(deltaEf / kappa) ** 2.) / step[1]
        elif option == 2:
            gS = 1. / (1. + (deltaSf / kappa) ** 2.) / step[0]
            gE = 1. / (1. + (deltaEf / kappa) ** 2.) / step[1]

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one
        # pixel. don't as questions. just do it. trust me.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        imgout += gamma * (NS + EW)

        if ploton:
            iterstring = "Iteration %i" % (ii + 1)
            ih.set_data(imgout)
            ax2.set_title(iterstring)
            fig.canvas.draw()
        # sleep(0.01)

    return imgout


def color_layer_segmentation(img):
    colors = np.unique(img)  # De mas oscuro a mas claro

    each_color_picture = []  # Con el valor a 1 y el resto a 0

    for value in colors:
        color_layer = copy.deepcopy(img)  # Con el valor a 1 y el resto a 0
        h = img.shape[0]
        w = img.shape[1]

        # iteramos sobre cada pixel
        for y in range(0, h):
            for x in range(0, w):
                if img[y, x] == value:
                    color_layer[y, x] = white
                else:
                    color_layer[y, x] = black

        each_color_picture.append(color_layer)

    return each_color_picture


def color_layer_segmantation_filled(img):
    colors = np.unique(img)  # De mas oscuro a mas claro

    each_filled_picture = []  # Con el valor y los menores al valor a 1 y el resto a 0

    for value in colors:
        filled_color_layer = copy.deepcopy(img)

        h = img.shape[0]
        w = img.shape[1]

        # iteramos sobre cada pixel
        for y in range(0, h):
            for x in range(0, w):
                if img[y, x] <= value:
                    filled_color_layer[y, x] = white
                else:
                    filled_color_layer[y, x] = black

        # Desechamos las imagenes que sean completamente negras
        if cv2.countNonZero(filled_color_layer) != (filled_color_layer.shape[0] * filled_color_layer.shape[1]):
            each_filled_picture.append(filled_color_layer)

    return each_filled_picture
