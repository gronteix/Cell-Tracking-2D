import numpy as np
import os
import pandas
import tqdm

# Functions to clean, filter data commonly used in the files analyzed


def _getDataFrame(path):

    resultFrame = pandas.DataFrame()

    for fileName in os.listdir(path):

        try:

            resultFrame = resultFrame.append(pandas.read_csv(os.path.join(path, fileName)))

        except:

            None

    return resultFrame

def _binExperiments(dataFrame, expToBin):

    for exp in expToBin:

        dataFrame = dataFrame[dataFrame['experiment'] != exp]

    return dataFrame

def _getVelocities(dataFrame, n):

    AnalysisFrame = pandas.DataFrame()

    for experiment in dataFrame['experiment'].unique():

        t = dataFrame.loc[dataFrame['experiment'] == experiment]

        for particle in t['particle'].unique():

            locFrame = t.loc[t['particle'] == particle]

            x = locFrame['x']
            xShift = x.shift(n)
            y = locFrame['y']
            yShift = y.shift(n)
            frame = locFrame['frame']
            frameShift = frame.shift(n)

            locFrame['dx'] = x - xShift
            locFrame['dy'] = y - yShift
            locFrame['dframe'] = frame - frameShift
            locFrame['dr'] = np.sqrt(locFrame['dx']**2 + locFrame['dy']**2)
            locFrame['v'] = np.sqrt(locFrame['dx']**2 + locFrame['dy']**2)/locFrame['dframe']

            AnalysisFrame = AnalysisFrame.append(locFrame)

    return AnalysisFrame

def _cutOutCells(dataFrame):

    """

    Remove cells too close to the border of the drop. Are stuck and risk
    falsifying the analysis.

    """

    wellDiameter = 440
    aspectRatio = 3

    cropDist = wellDiameter*aspectRatio

    dataFrame['x Loc'] = dataFrame['x pos'] - cropDist/2
    dataFrame['y Loc'] = dataFrame['y pos'] - cropDist/2

    r = np.sqrt(dataFrame['x Loc']**2 + dataFrame['y Loc']**2)

    return dataFrame[r < cropDist/2 - 100]

def _distanceClean(t, minDist):

    # and whose total travel distance is longer than approx 200 px
    # we use the sum function to take away the bias from a moving frame

    for particle in t['particle'].unique():

        if len(t[t['particle'] == particle]) > 2:

            Dx = t.loc[t['particle']==particle, 'dx'].sum()
            Dy = t.loc[t['particle']==particle, 'dy'].sum()

            if np.sqrt((Dx)**2+(Dy)**2) < minDist:

                t = t[t.particle != particle]

        else:

            t = t[t.particle != particle]

    return t

def _loopThru(tTot, minDist):

    CleanFrame = pandas.DataFrame()

    TempFrame = _cutOutCells(tTot)

    for experiment in TempFrame['experiment'].unique():

        t = TempFrame[TempFrame['experiment'] == experiment]

        t = _getVelocities(t, 1)
        t = _distanceClean(t, 20)

        CleanFrame = CleanFrame.append(_distanceClean(t, minDist), ignore_index=True)

    return CleanFrame

def _getArrest(df):

    """

    get arrest phases of cell. Returns a data farme object.

    """

    for ID in df['ID'].unique():

        lf = df[df['ID'] == ID]

        for particle in lf['particle'].unique():

            sf = lf[lf['particle'] == particle]

            df.loc[(df['particle'] == particle) &
                   (df['ID'] == ID),
                   'v gauss'] = scipy.ndimage.filters.gaussian_filter1d(sf['v'], sigma=5, mode = 'constant')

    for ind, row in df.iterrows():

        if row['v gauss'] < 2.2:
            df.loc[ind, 'arrest'] = 'arrest'

        else:
            df.loc[ind, 'arrest'] = 'movement'

    return df

def _getTrackProperties(dataFrame):

    dataFrame = _getArrest(dataFrame)

    HitFrame = pandas.DataFrame()
    i = 0

    for ID in dataFrame['ID'].unique():

        df = dataFrame[dataFrame['ID'] == ID]

        for particle in df['particle'].unique():

            lf = df[df['particle'] == particle]
            lf = lf.sort_values('frame')

            if (len(lf) > 10):

                HitFrame.loc[i, 'particle'] = particle
                HitFrame.loc[i, 'ID'] = ID
                HitFrame.loc[i, '$<v>$'] = lf['v'].mean()
                HitFrame.loc[i, 'arrest %'] = len(lf[lf['arrest'] == 'arrest'])/len(lf)
                i += 1

    return HitFrame
