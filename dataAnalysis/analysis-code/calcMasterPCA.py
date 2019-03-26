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
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline, Pipeline
#  load options
from currentExperiment import *
from joblib import dump, load

masterSpikeMat = preproc.loadSpikeMats(
    binnedSpikePath, rasterOpts, loadAll=True)[0][0]
    
nComp = 5
pca = PCA(n_components=nComp)
compNames = ['PC{}'.format(i+1) for i in range(nComp)]
estimator = Pipeline([('dimred', pca)])
features = estimator.fit_transform(masterSpikeMat.values)
featuresDF = pd.DataFrame(
    features, index=masterSpikeMat.index, columns=compNames)
# !!! loadSpikemats only does the first segment right now
chooseTDChans = ['ins_td0', 'position', 'amplitude']
masterPosMat = preproc.loadSpikeMats(
    experimentDataPath, rasterOpts,
    loadAll=True, chans=chooseTDChans)[0][0]
masterPosMat.rename(
    columns={
        'ins_td0': 'ins_td0',
        'position': 'position',
        'amplitude': 'tdAmplitude'
    }, inplace=True
)
#  INS amplitudes are in 100s of uA
masterPosMat['tdAmplitude'] = masterPosMat['tdAmplitude'] / 10

featuresDF = pd.concat((
    featuresDF, masterPosMat
    ), axis=1)

featuresDF.to_hdf(masterFeaturePath, 'features')
dump(estimator, estimatorPath)