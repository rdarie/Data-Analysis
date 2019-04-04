import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
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
from joblib import dump, load
import quantities as pq
#  all experimental days
dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToPlot = ['PC1#0', 'PC2#0']
waveformsDict = {}
for segIdx, dataSeg in enumerate(dataBlock.segments):
    stProxys = dataSeg.filter(objects=SpikeTrainProxy)
    spiketrains = [
        stP.load(load_waveforms=True)
        for stP in stProxys
        if stP.name in chansToPlot]
    spiketrains = preproc.loadContainerArrayAnn(
        trainList=spiketrains)
    waveformsDF = spikeTrainWaveformsToDF(spiketrains)
    waveformsDict.update({segIdx: waveformsDF})

allWaveforms = pd.concat(
    waveformsDict,
    names=['segment'] + waveformsDF.index.names)
allWaveforms.columns.name = 'bin'
allWaveforms = allWaveforms.stack().reset_index(name='signal')

dataQuery = '&'.join([
    '(RateInHz >= 100)',
    '(feature==\'PC1#0\')',
    '((pedalSizeCat == \'M\') | (pedalSizeCat == \'Control\'))',
    ])
plotDF = allWaveforms.query(dataQuery)

g = sns.relplot(
    x='bin', y='signal',
    hue='amplitudeFuzzy',
    col='pedalMovementCat', row='program',
    height=5, aspect=.7, kind='line', data=plotDF)
plt.show()