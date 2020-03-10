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
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import fit_grid_point
from sklearn.metrics import mean_squared_error, r2_score
from collections import Iterable
#  load options
from currentExperiment import *
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

#  all experimental days
experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_analyze.nix')
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    trialFilesStim['ins']['experimentName'] + '_binarized.nix')
'''
experimentDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    'Block001_analyze.nix')
binnedSpikePath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    'Block001_binarized.nix')
'''
dataBlock = preproc.loadWithArrayAnn(
    experimentDataPath, fromRaw=False)
#  passing binnedspikepath interpolates down to 1 msec
tdDF, stimStatus = preproc.unpackAnalysisBlock(
    dataBlock, interpolateToTimeSeries=True,
    binnedSpikePath=binnedSpikePath)
#  todo: get from binnedSpikes

confirmationPlot = True

tStart = 325
tStop = 365
sns.set()
progAmpNames = rcsa_helpers.progAmpNames
tdMask = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)
fig, ax = plt.subplots(2, 1)
for pC in progAmpNames:
    plotTrace = tdDF.loc[tdMask, pC] / 10
    ax[0].plot(tdDF.loc[tdMask, 't'] - tStart, plotTrace, label=pC)
ax[0].legend(['Rostral', 'Caudal', 'Caudal Midline', 'No Stim'])
ax[0].set_ylabel('Amplitude (mA)')
ax[0].set(xticklabels=[])
ax[1].plot(tdDF.loc[tdMask, 't'] - tStart, tdDF.loc[tdMask, 'position'], label='position')
ax[1].legend()
ax[1].set_ylabel('Position (deg.)')
ax[1].set_xlabel('Time (sec)')
plt.show()
