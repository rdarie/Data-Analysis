import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Iterable
#  load options
from currentExperiment import *
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
from collections import OrderedDict

#plt.style.use("dark_background")

sns.set_context("talk")
unpackedFeatures = pd.read_hdf(masterFeaturePath, 'features')
unpackedFeatures['position'] = (
    unpackedFeatures['position'] -
    unpackedFeatures['position'].max()) * 100
unpackedFeatures['position'] = unpackedFeatures['position'].abs()

pcList = ['PC1', 'PC2', 'PC3']
#unpackedFeatures.loc[:, pcList] = hf.filterDF(
#    unpackedFeatures.loc[:, pcList], 100, lowPass=30)

if True:
    fig, ax = plt.subplots(2, 1, sharex=True)
    ax[0].plot(unpackedFeatures['position'] / 100, label='position (deg.)')
    ax[0].plot(unpackedFeatures['tdAmplitude'], label='amplitude (mA)')
    ax[0].legend()
    ax[1].plot(unpackedFeatures[pcList], label='PC (a.u)')
    ax[1].set_xlabel('time (sec.)')
    ax[1].legend(); plt.show()
    #  plt.plot(unpackedFeatures.query(dataQuery)['position']); plt.show()
winWidth = 40
nFrames = 999
dataQuery = '&'.join([
    '(bin > 352)',
    '(bin < 362)'
    ])

sns.set_style("darkgrid", {'axes.facecolor': '0.7', 'figure.facecolor': '0.7'})
plotKws = {'linewidth': 2}
fig = plt.figure(figsize=(6, 4), dpi=300, tight_layout={'h_pad': 0, 'w_pad': 1.5})
featAx = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d', aspect='auto')
moveAx = plt.subplot2grid((1, 3), (0, 2), projection='3d', aspect='equal')

aniFeat = hf.animateDFSubset3D(
    unpackedFeatures, dataQuery, winWidth, nFrames,
    xyzList=pcList, pltKws=plotKws, ax=featAx,
    colorCol='tdAmplitude')
plotKws = {'linewidth': 2}
aniPedal = hf.animateAngle3D(
    unpackedFeatures, dataQuery, winWidth, nFrames, ax=moveAx,
    pltKws=plotKws, saveToFile='', extraAni=[aniFeat])
plt.show()
# move.mp4

dataQuery = '&'.join([
    '(bin > 429)',
    '(bin < 450)'
    ])

nFrames = 2000
sns.set_style("darkgrid", {'axes.facecolor': '0.7', 'figure.facecolor': '0.7'})
fig = plt.figure(figsize=(6, 4), dpi=300, tight_layout={'h_pad': 0, 'w_pad': 1.5})

featAx = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d', aspect='auto')
moveAx = plt.subplot2grid((1, 3), (0, 2), projection='3d', aspect='equal')

aniFeat = hf.animateDFSubset3D(
    unpackedFeatures, dataQuery, winWidth, nFrames, ax=featAx,
    xyzList=['PC1', 'PC2', 'PC3'], pltKws=plotKws,
    colorCol='tdAmplitude', saveToFile='')
aniPedal = hf.animateAngle3D(
    unpackedFeatures, dataQuery, winWidth, nFrames, ax=moveAx,
    pltKws=plotKws, saveToFile='', extraAni=[aniFeat])
plt.show()