import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, Epoch, AnalogSignal, SpikeTrain)
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
import quantities as pq
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
from neo import (
    Unit, AnalogSignal, Event, Epoch,
    Block, Segment, ChannelIndex, SpikeTrain)
#  load options
from currentExperiment import *

trialList = [1, 2, 3, 4]
suffixList = ['_binarized', '_analyze']
#  suffixList = ['_analyze']
#  suffixList = ['_binarized']
for suffix in suffixList:
    experimentDataPath = os.path.join(
        trialFilesStim['ins']['folderPath'],
        trialFilesStim['ins']['experimentName'],
        trialFilesStim['ins']['experimentName'] + suffix + '.nix')
    for idx, trialIdx in enumerate(trialList):
        #  trialIdx = 2
        #  idx = 0
        #  trialIdx = 3
        #  idx = 1
        trialDataPath = os.path.join(
            trialFilesStim['ins']['folderPath'],
            trialFilesStim['ins']['experimentName'],
            'Trial00{}'.format(trialIdx) + suffix + '.nix')
        if idx == 0:
            masterBlock = preproc.loadWithArrayAnn(
                trialDataPath, fromRaw=False)
            #  pdb.set_trace()
            if suffix == '_analyze':
                for st in masterBlock.filter(objects=SpikeTrain):
                    st.unit.name = st.name
                    st.unit.channel_index.name = st.name
                for asig in masterBlock.filter(objects=AnalogSignal):
                    asig.channel_index.name = asig.name
            elif suffix == '_binarized':
                for seg in masterBlock.segments:
                    seg.spiketrains = []
                for asig in masterBlock.filter(objects=AnalogSignal):
                    asig.name = asig.channel_index.name
            masterTStart = masterBlock.filter(objects=AnalogSignal)[0].t_start
            oldTStop = masterBlock.filter(objects=AnalogSignal)[0].t_stop
            typesNeedRenaming = [Segment, SpikeTrain, AnalogSignal, Event]
            for objType in typesNeedRenaming:
                for child in masterBlock.filter(objects=objType):
                    child.name = 'seg{}_{}'.format(idx, child.name)
        else:
            dataBlock = preproc.loadWithArrayAnn(
                trialDataPath, fromRaw=False)
            if suffix == '_analyze':
                for st in dataBlock.filter(objects=SpikeTrain):
                    st.unit.name = st.name
                    st.unit.channel_index.name = st.name
                for asig in dataBlock.filter(objects=AnalogSignal):
                    asig.channel_index.name = asig.name
            elif suffix == '_binarized':
                for seg in dataBlock.segments:
                    seg.spiketrains = []
                for asig in dataBlock.filter(objects=AnalogSignal):
                    asig.name = asig.channel_index.name
            tStart = dataBlock.filter(objects=AnalogSignal)[0].t_start
            timeOffset = oldTStop - tStart
            dataBlock = hf.timeOffsetBlock(dataBlock, timeOffset, masterTStart)
            #  [i.times for i in dataBlock.filter(objects=SpikeTrain)]
            #  [i.unit.channel_index.name for i in masterBlock.filter(objects=SpikeTrain)]
            tStop = dataBlock.filter(objects=AnalogSignal)[0].t_stop
            for objType in typesNeedRenaming:
                for child in dataBlock.filter(objects=objType):
                    child.name = 'seg{}_{}'.format(idx, child.name)
            masterBlock.merge(dataBlock)
            oldTStop = tStop
    masterBlock = preproc.purgeNixAnn(masterBlock)
    writer = neo.io.NixIO(filename=experimentDataPath)
    writer.write_block(masterBlock)
    writer.close()
