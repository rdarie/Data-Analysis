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
    filename=experimentTriggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

chansToPlot = np.unique([
    i.name
    for i in dataBlock.filter(objects=SpikeTrainProxy)
    if '_fr' in i.name])
#  chansToPlot = ['PC{}#0'.format(i+1) for i in range(10)]
#  chansToPlot = ['PC3#0']
for chanName in chansToPlot:
    waveformsDict = {}
    for segIdx, dataSeg in enumerate(dataBlock.segments):
        stProxys = dataSeg.filter(objects=SpikeTrainProxy)
        spiketrains = [
            stP.load(load_waveforms=True)
            for stP in stProxys
            if chanName in stP.name]
        spiketrains = preproc.loadContainerArrayAnn(
            trainList=spiketrains)
        waveformsDF = preproc.spikeTrainWaveformsToDF(spiketrains)
        waveformsDict.update({segIdx: waveformsDF})
    allWaveforms = pd.concat(
        waveformsDict,
        names=['segment'] + waveformsDF.index.names)
    allWaveforms.columns.name = 'bin'
    
    allWaveformsPlot = allWaveforms.stack().reset_index(name='signal')
    dataQuery = '&'.join([
        '(RateInHz >= 100)',
        '(feature==\'{}\')',
        '((pedalSizeCat == \'M\') | (pedalSizeCat == \'Control\'))',
        ]).format(chanName)
    plotDF = allWaveformsPlot.query(dataQuery)
    g = sns.relplot(
        x='bin', y='signal',
        hue='amplitudeFuzzy',
        col='pedalMovementCat', row='program',
        height=5, aspect=1.2, kind='line', data=plotDF)
    plt.suptitle(chanName)
    #  plt.show()
    #  block=False
    #  plt.pause(3)
    plt.savefig(os.path.join(figureFolder, '{}.pdf'.format(chanName)))
    plt.close()
