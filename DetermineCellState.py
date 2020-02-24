import os
import sys
import numpy as np
import pandas
from numpy import unravel_index
import pims
import matplotlib.pyplot as plt

import SpheroidSegmentBF

def _getState(pos, xSph, ySph, radius):

    """

    Returns:
     - 0 if cell not on spheroid
     - 1 if cell on spheroid

    """

    if not isinstance(pos, tuple):

        return print("position not tuple")

    x, y = pos

    if any((xSph-x)**2 + (ySph-y)**2 < radius**2):

        return 1

    else: return 0

def _selectCells(cellFrame, experiment, IMAGECHANNEL, PATH,
                      maskSize, wellSize, aspectRatio):

    img = pims.ImageSequence(os.path.join(PATH, experiment, IMAGECHANNEL, '*.tif'), as_grey=True)

    im = img[np.min(cellFrame['frame'].dropna())]
    xCenter, yCenter = SpheroidSegmentBF._getCenter(im,maskSize,wellSize,aspectRatio)

    cellFrame['xCenter'] = xCenter
    cellFrame['yCenter'] = yCenter

    areaToKeep = maskSize*aspectRatio
    cellFrame = cellFrame[(cellFrame['x']-xCenter)**2 + (cellFrame['y']-yCenter)**2 < areaToKeep**2]

    return cellFrame

def _loopThroughCells(cellFrame, radius, experiment, IMAGECHANNEL, PATH,
                      wellDiameter, marginDistance, aspectRatio):

    #initialize all states to zero
    cellFrame['state'] = 0

    loc = cellFrame[cellFrame['frame'] == np.min(cellFrame['frame'])]
    xCenter = loc['xCenter'].iloc[0]
    yCenter = loc['yCenter'].iloc[0]

    xSph, ySph = SpheroidSegmentBF._getSphCoords(PATH, experiment,
                            str(np.min(cellFrame['frame'])), IMAGECHANNEL,
                            wellDiameter, marginDistance, aspectRatio)


    for frame in sorted(cellFrame['frame'].unique()):

        loc = cellFrame[cellFrame['frame'] == frame]
        xCenter = loc['xCenter'].iloc[0]
        yCenter = loc['yCenter'].iloc[0]

        for particle in loc['particle'].unique():

            temp = loc[loc['particle'] == particle]
            cropDist = wellDiameter*aspectRatio
            x = temp['x'].iloc[0] - yCenter + cropDist/2
            y = temp['y'].iloc[0] - xCenter + cropDist/2

            #x = temp['x'].iloc[0]
            #y = temp['y'].iloc[0]

            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'state'] = _getState((y,x), xSph, ySph, radius)
            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'ySph'] = np.mean(xSph)
            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'xSph'] = np.mean(ySph)

            #counting the number of px belonging to sph
            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'Spheroid area'] = len(xSph)

            # reference change between imshow and scatter (xy axis not the same)
            # needed to switch to get the right results.

            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'xLocalRef'] = x
            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'yLocalRef'] = y

            cellFrame.loc[(cellFrame['frame'] == frame) &
                (cellFrame['particle'] == particle), 'sph Radius'] = np.sqrt(len(xSph/np.pi))

        if frame % 60 == 0:

            _verifyCellState(cellFrame, PATH, experiment, IMAGECHANNEL, frame, wellDiameter,
                                    aspectRatio, marginDistance)


            #on fait une image sur 20 pour limiter le nombre de points de temps

            xSphNew, ySphNew = SpheroidSegmentBF._getSphCoords(PATH, experiment,
                            str(frame), IMAGECHANNEL,
                            wellDiameter, marginDistance, aspectRatio)

            if len(xSphNew) > 0:

                xSph = xSphNew
                ySph = ySphNew

            # on fait le plot test

    return cellFrame

def _verifyCellState(cellFrame, PATH, experiment, IMAGECHANNEL, frame, wellDiameter,
                        aspectRatio, marginDistance):

    if not os.path.exists(os.path.join(PATH, experiment, 'CD8 on spheroid test')):
        os.makedirs(os.path.join(PATH, experiment, 'CD8 on spheroid test'))

    savePath = os.path.join(PATH, experiment, 'CD8 on spheroid test')

    img = pims.ImageSequence(os.path.join(PATH, experiment, IMAGECHANNEL, '*.tif'), as_grey=True)
    img = img[frame]

    cropDist = wellDiameter*aspectRatio

    loc = cellFrame[cellFrame['frame'] == frame]
    xCenter = loc['xCenter'].iloc[0]
    yCenter = loc['yCenter'].iloc[0]

    i = SpheroidSegmentBF._crop(img, img, (xCenter, yCenter), wellDiameter, wellDiameter, aspectRatio)
    r = SpheroidSegmentBF._cropper(img, (xCenter, yCenter), wellDiameter, marginDistance, aspectRatio)
    rRegion = SpheroidSegmentBF._findSpheroid(r, wellDiameter, aspectRatio, marginDistance)

    x = loc['x'] - yCenter + cropDist/2
    y = loc['y'] - xCenter + cropDist/2

    fig, ax = plt.subplots(1,1, figsize = (5,5))
    plt.imshow(i, cmap='gray', origin = 'lower')
    plt.imshow(rRegion, alpha = 0.1, origin = 'lower')
    plt.scatter(x, y, c = loc['state'], label = loc['state'])

    for particle in loc['particle'].unique():

        state = loc.loc[loc['particle'] == particle, 'state'].iloc[0]
        xplot = loc.loc[loc['particle'] == particle, 'x'].iloc[0] - yCenter + cropDist/2
        yplot = loc.loc[loc['particle'] == particle, 'y'].iloc[0] - xCenter + cropDist/2
        plt.text(xplot, yplot, state)

    plt.savefig(os.path.join(savePath, 'testFrame_' + str(frame) +'.jpeg'))
    plt.legend()
    plt.close(fig)
    return


def _loopThroughExperiments(PATH, DATAPATH, SAVEPATH, CHANNEL, wellDiameter, marginDistance, aspectRatio):

    for experiment in os.listdir(DATAPATH):

        cellFrame = pandas.read_csv(os.path.join(DATAPATH, experiment))

        if len(cellFrame) > 0:

            cellFrame = _selectCells(cellFrame, experiment, CHANNEL, PATH,
                          wellDiameter, wellDiameter, aspectRatio)

            cellFrame = _loopThroughCells(cellFrame, 10, experiment, CHANNEL, PATH,
                                  wellDiameter, marginDistance, aspectRatio)

            if not os.path.exists(SAVEPATH):
                os.makedirs(SAVEPATH)

            cellFrame.to_csv(os.path.join(SAVEPATH, experiment))

    return
