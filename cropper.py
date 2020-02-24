import numpy as np
import pims
from numpy import unravel_index
import cv2
import os
from tqdm import tqdm_notebook as tqdm
import skimage



def _crop(imgToCrop,imgMask,maskSize,wellSize,aspectRatio):

    """Crop function. Works only on 2D images.
    """

    (xc, yc) = _getCenter(imgMask,maskSize,wellSize,aspectRatio)

    cropDist = maskSize*aspectRatio

    startx = max(xc-(cropDist//2), 0)
    starty = max(yc-(cropDist//2), 0)

    return imgToCrop[int(starty):int(starty+cropDist),int(startx):int(startx+cropDist)]

def _makeCircMask(maskSize, wellSize, aspectRatio):

    cropDist = maskSize*aspectRatio

    X = np.arange(0, cropDist)
    Y = np.arange(0, cropDist)
    X, Y = np.meshgrid(X, Y)

    mask = ((np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) > (wellSize*aspectRatio)//2 - 10*aspectRatio) &
            (np.sqrt((X-cropDist//2)**2 + (Y-cropDist//2)**2) < (wellSize*aspectRatio)//2 + 10*aspectRatio))

    return mask.astype(np.int)

def _getCenter(imgMask,maskSize,wellSize,aspectRatio):

    mask = _makeCircMask(maskSize,wellSize,aspectRatio)

    conv = cv2.filter2D(imgMask, cv2.CV_32F, mask)

    return unravel_index(conv.argmin(), conv.shape)

def _cropAll(PATH,maskSize,wellSize,aspectRatio):

    if not os.path.exists(PATH + r'\cropped'):
        os.mkdir(PATH + r'\\' + 'cropped')

    img = pims.ImageSequence(PATH + '\\*.tif', as_grey=True)
    i = 0

    for im in tqdm(img):

        skimage.external.tifffile.imsave(PATH + r'\\cropped\\' + 'crop_%d.tif' %i,
            _crop(im, im,maskSize,wellSize,aspectRatio))
        i += 1

    return
