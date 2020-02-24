import multiprocessing as mp
import os
import glob
import tqdm

import numpy as np
import pandas
from pandas import DataFrame, Series  # for convenience
import scipy.misc
import pims
import trackpy as tp
from tqdm import tqdm_notebook as tqdm
import glob
import os
from skimage import io
import pandas as pd

###### PARAMETERS ######


PATH = r'D:\20200127'
sizeReal = 12
minmass = 700
distanceReal = 20
pxtoum = 0.66
sep = 15

distance = distanceReal/pxtoum
size = ((sizeReal/pxtoum)//2)*2+1
separation = ((sep/pxtoum)//2)*2+1

###### START OF STUDY ######

def _runThru(path, size, minmass, distance, folder):

    print(path)

    frames = pims.ImageSequence(path + '\\2\\*.tif', as_grey=True)
    f = tp.batch(frames, size, separation = separation, invert=False,
                    minmass = minmass)
    t = tp.link_df(f, distance, memory=2)
    t = tp.filter_stubs(t, 1)

    t['experiment'] = folder
    t['raw images'] = path + '\\2'

    directory = PATH + r'\\DataFrames'

    if not os.path.exists(directory):
        os.makedirs(directory)
    t.to_csv(directory + r'\\' + folder)

# process._sortFiles(path)

if __name__ == '__main__':

    print('Start analysis')

    output = mp.Queue()

    processes = [mp.Process(target=_runThru, args=(PATH + r'\\' + folder,
        size, minmass, distance, folder)) for folder in os.listdir(PATH)]

    for p in processes:
        p.start()

    for p in tqdm(processes):
        p.join()
