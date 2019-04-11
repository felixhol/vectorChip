'''
author: Felix Hol
date: 2019 April 10
content: code to track mosquitoes, several filtering parameters will need tweaking depending on imaging parameters.
'''

import numpy as np
import matplotlib as mpl
mpl.use('TkAgg') ### this is a workaround for a bug when running on macOS, may not be necessary anymore
import matplotlib.pyplot as plt
import os
import itertools as it
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import skimage
from skimage import data, io, util
from skimage.feature import (match_descriptors, peak_local_max, match_template, corner_peaks, corner_harris, plot_matches, BRIEF)
from skimage.color import rgb2gray
from skimage.draw import circle
from skimage.measure import label, regionprops
from skimage.morphology import binary_dilation, erosion, dilation, opening, binary_closing, closing, white_tophat, remove_small_objects, disk, black_tophat, skeletonize, convex_hull_image
from scipy import ndimage as ndi
import trackpy as tp
import pylab
import math
from joblib import Parallel, delayed
import multiprocessing
from datetime import datetime
from tqdm import trange
import pickle

dataDir = '/Users/felix/Documents/mosquitoes/mosquitoData/biteData/190401_01/01'
saveDir = '/Users/felix/Documents/mosquitoes/mosquitoData/biteData/'
frames = pims.ImageSequence(dataDir+'/*.png', as_grey=True)

####create background image

start = 10
stop = len(frames) - 100
step = math.floor(len(frames)/25)

numBGframes = int(np.ceil((stop - start) /step) + 1)
frameSize = frames[1].shape
BG = np.zeros([frameSize[0], frameSize[1], numBGframes])

j = 1
for i in range(start, stop, step):
    j += 1
    BG[:, :, j - 1] = np.invert(frames[i])    

BG = np.median(BG, axis=2)

#### get centroid and head coordinates

def trackMosq2(i):
    selem1 = disk(8)
    selem2 = disk(1)
    A = np.zeros(frameSize)
    A = A + np.invert(frames[i])
    B = A - BG
    if B.min() > 0:
        Bm = B - B.min()
    else:
        Bm = B
    Bt = Bm > 50
    Bts = remove_small_objects(Bt, min_size=300)
    Be = erosion(Bts, selem2)
    Bf = remove_small_objects(Be, min_size=200)
    Bc = binary_closing(Bf, selem1)
    C = B * Bc
    eroded = erosion(C, selem2)
    eroded = skimage.filters.gaussian(eroded, 4)
    eroded[eroded < 0] = 0
    erL = label(eroded>0)
    erR = regionprops(erL, C, coordinates='xy')
    l = 1
    for props in erR:
        if props.area > 10000:
            erL[erL==l] = 0
        if props.major_axis_length > 500:
            erL[erL==l] = 0
        l = l +1
    erLf = label(erL>0)
    erodedF = eroded * (erLf > 0)
    erRf = regionprops(erLf, C, coordinates='xy')
    centroids = np.zeros([len(erRf), 2])
    numCent = 0
    for props in erRf:
        centroids[numCent] = props.centroid
        numCent += 1
    coordinates = peak_local_max(eroded, min_distance=20, exclude_border=1)
    cS= coordinates.shape; numCoor = cS[0]
    cenS= centroids.shape; numCen = cenS[0]
    frameNo = i
    frameNoA = np.zeros((numCoor,1), dtype=np.int)
    frameNoCen = np.zeros((numCen,1), dtype=np.int)
    frameNoA[:] = frameNo
    frameNoCen[:] = frameNo
    coordinatesF = np.hstack((coordinates,frameNoA))
    centroidsF = np.hstack((centroids,frameNoCen))
    coordinatesF = np.hstack((coordinates,frameNoA))
    numCoords = coordinatesF.shape[0]
    numCents = centroidsF.shape[0]
    return centroidsF, coordinatesF, numCents, numCoords, 


num_cores = multiprocessing.cpu_count()
results = Parallel(n_jobs=num_cores)(delayed(trackMosq2)(i) for i in trange(1, 100))

centroidsAllT = np.zeros((1,3))
coordinatesAllT = np.zeros((1,3))

for i in range(len(results)):
    centroidsAllT = np.vstack((centroidsAllT,results[i][0]))
    coordinatesAllT = np.vstack((coordinatesAllT,results[i][1]))

os.chdir(saveDir)
with open('190402_1-100_centroidsAndCoordinates01.pkl', 'wb') as f:
    pickle.dump(results, f)


