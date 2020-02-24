import os
import sys
import tqdm
import numpy as np
import pandas

from tqdm import tqdm_notebook as tqdm
import pims
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt

import cv2
from numpy import unravel_index

from scipy import ndimage, misc
from skimage.measure import label, regionprops


def _makeDiskMask(maskSize, wellSize, aspectRatio):

    """

    Makes a circular mask of given radius.

    """

    cropDist = maskSize*aspectRatio

    X = np.arange(0, cropDist)
    Y = np.arange(0, cropDist)
    X, Y = np.meshgrid(X, Y)

    mask = (np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) < (wellSize*aspectRatio)//2 + 20*aspectRatio)

    return mask.astype(np.int)

def _makeCircMask(maskSize, wellSize, aspectRatio):

    """

    Makes a disk mask of given radius. Used for finding well center.

    """

    cropDist = maskSize*aspectRatio
    X = np.arange(0, cropDist)
    Y = np.arange(0, cropDist)
    X, Y = np.meshgrid(X, Y)

    mask = ((np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) > (wellSize*aspectRatio)//2 - 30*aspectRatio) &
            (np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) < (wellSize*aspectRatio)//2 + 20*aspectRatio))

    return mask.astype(np.int)

def _getCenter(imgMask,maskSize,wellSize,aspectRatio):

    """

    Find the well center.

    """

    mask = _makeCircMask(maskSize,wellSize,aspectRatio)

    conv = cv2.filter2D(imgMask, cv2.CV_32F, mask, borderType = cv2.BORDER_REPLICATE)

    x, y = unravel_index(conv.argmin(), conv.shape)
    a, b = np.shape(conv)

    if not ((np.abs(x-a/2) < a/4) & (np.abs(y-a/2) < a/4)):
        x, y = unravel_index(conv.argmax(), conv.shape)

    return x, y

def _crop(imgToCrop,imgMask, wellCenter, maskSize,wellSize,aspectRatio):

    """Crop function. Works only on 2D images.

     - imgToCrop: image to be cropped
     - imgMask: image used to select where to crop
    """

    xc, yc = wellCenter

    cropDist = maskSize*aspectRatio
    startx = max(xc-(cropDist//2), 0)
    starty = max(yc-(cropDist//2), 0)

    return imgToCrop[int(startx):int(startx+cropDist),int(starty):int(starty+cropDist)]

def _cropper(imToCrop, wellCenter, wellDiameter, marginDistance, aspectRatio):

    """

    Function to take image, find the well center,
    throw away all the points beyond the marginDistance.

    """

    img = _crop(imToCrop, imToCrop, wellCenter, wellDiameter, wellDiameter, aspectRatio)

    mask = _makeDiskMask(wellDiameter, wellDiameter-marginDistance, aspectRatio)
    t = np.zeros(np.shape(mask))
    a, b = np.shape(img)
    t[:a, :b] = img
    #combine pour gerer les crops de forme non carree

    finalIm = np.multiply(t, mask)
    finalIm[finalIm == 0] = np.max(finalIm)

    return gaussian_filter(finalIm, sigma=5)


def _findSpheroid(imCropped, wellDiameter, aspectRatio, marginDistance, fraction = 5,
                  minRegionArea = 15000, maxRegionArea = 120000):

    """

    We find the spheroid by thresholding the intensity
    and area filling. Sph. must have a dark border around
    it.

    """


    result1 = ndimage.sobel(imCropped, 1)
    result2 = ndimage.sobel(imCropped, 0)

    mask = _makeDiskMask(wellDiameter, wellDiameter-marginDistance-20, aspectRatio)
    sobelMasked = np.multiply(mask, np.sqrt(result1**2+result2**2))
    toThresh = gaussian_filter(sobelMasked, sigma=5)

    imThresh = toThresh > np.max(toThresh)/fraction

    cnts, h = cv2.findContours(imThresh.astype('uint8'), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    temp = cv2.drawContours(imThresh.astype('uint8'), cnts, -1, (255,255,255), thickness=cv2.FILLED)

    imLabel = label(temp)

    for region in regionprops(imLabel):

        if region.area < minRegionArea:
        #check it is inside or outside

            temp[imLabel == region.label] = 0
            #region given same value as sph. border

        if region.area > maxRegionArea:
        #check it is inside or outside

            temp[imLabel == region.label] = 0
            #region given same value as sph. border

        if region.eccentricity > 0.8:
        #check it is inside or outside

            temp[imLabel == region.label] = 0
            #region given same value as sph. border

    return temp

def _verifySegmentation(BFimage, rRegion, PATH, experiment, time):

    if not os.path.exists(os.path.join(PATH, experiment, 'Spheroid Region Detection')):
        os.makedirs(os.path.join(PATH, experiment, 'Spheroid Region Detection'))
    savePath = os.path.join(PATH, experiment, 'Spheroid Region Detection')

    fig, ax = plt.subplots(1,1, figsize = (10,10))

    plt.imshow(BFimage, cmap='gray')
    plt.imshow(rRegion, alpha = 0.1)
    plt.savefig(os.path.join(savePath, 'testFrame_%frame.jpeg' %round(int(time,0))))
    plt.close(fig)

    return

def _getSphCoords(PATH, experiment, time, CHANNEL, wellDiameter, marginDistance, aspectRatio):

    """
    CORE FUNCTION:

    Function to retrieve the spheroid coordinates from the BF images. Relies
    upon ID by max gradient values.

    """

    img = pims.ImageSequence(os.path.join(PATH, experiment, CHANNEL, '*.tif'), as_grey=True)
    imToCrop = img[int(time)]

    xCenter, yCenter = _getCenter(img[0],wellDiameter,wellDiameter,aspectRatio)

    BFimage = _cropper(imToCrop, (xCenter, yCenter), wellDiameter, marginDistance, aspectRatio)
    rRegion = _findSpheroid(BFimage, wellDiameter, aspectRatio, marginDistance)


    # Image the segmentation to keep intermediary result of the segmentation.
    _verifySegmentation(BFimage, rRegion, PATH, experiment, time)

    return np.nonzero(rRegion)
