"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --window=window                        process with short window? [default: long]
    --lazy                                 load from raw, or regular? [default: False]
    --plotting                             load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --selectionName=selectionName          name in h5 for the saved data
    --unitQuery=unitQuery                  how to restrict channels? [default: fr_sqrt]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: midPeak]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --iteratorSuffix=iteratorSuffix        filename for cross_val iterator
    --selector=selector                    filename if using a unit selector
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('QT5Agg')   # generate postscript output
# matplotlib.use('Agg')   # generate postscript output
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
sns.set(
    context='talk', style='dark',
    palette='dark', font='sans-serif',
    font_scale=1.5, color_codes=True)

import os
import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import pdb
import numpy as np
import pandas as pd
import dataAnalysis.preproc.ns5 as ns5
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.covariance import ShrunkCovariance, LedoitWolf, EmpiricalCovariance
from sklearn.model_selection import cross_val_score, cross_validate, GridSearchCV
import joblib as jb
import dill as pickle
import gc
from docopt import docopt
import traceback

idxSl = pd.IndexSlice
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
analysisSubFolder, alignSubFolder = hf.processSubfolderPaths(
    arguments, scratchFolder)
dataFramesFolder = os.path.join(
    alignSubFolder, 'dataframes'
    ) # must exist from previous call to calcTestTrainSplit.py

cvIteratorSubfolder = os.path.join(
    scratchFolder, 'testTrainSplits',
    arguments['alignFolderName'])

if arguments['iteratorSuffix'] is not None:
    iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
else:
    iteratorSuffix = ''
iteratorPath = os.path.join(
    cvIteratorSubfolder,
    '{}_{}_{}{}_cvIterators.pickle'.format(
        blockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix))

with open(iteratorPath, 'rb') as f:
    loadingMeta = pickle.load(f)
iteratorsBySegment = loadingMeta.pop('iteratorsBySegment')
cv_kwargs = loadingMeta.pop('cv_kwargs')

listOfDataFrames = []

triggeredPath = os.path.join(
    alignSubFolder,
    blockBaseName + '{}_{}.nix'.format(
        inputBlockSuffix, arguments['window']))
alignedAsigsKWargs = loadingMeta.pop('alignedAsigsKWargs')
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']
if arguments['verbose']:
    prf.print_memory_usage('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
nSeg = len(dataBlock.segments)
for segIdx in range(nSeg):
    if arguments['verbose']:
        prf.print_memory_usage('extracting data on segment {}'.format(segIdx))
    if 'listOfROIMasks' in loadingMeta:
        alignedAsigsKWargs.update({'finalIndexMask': loadingMeta['listOfROIMasks'][segIdx]})
    aakwa = alignedAsigsKWargs.copy()
    if 'spectral' not in inputBlockSuffix:
        # needs downsampling
        aakwa['decimate'] = 20
        aakwa['rollingWindow'] = 200
    dataDF = ns5.alignedAsigsToDF(
        dataBlock,
        whichSegments=[segIdx],
        **aakwa)
    if 'listOfExampleIndexes' in loadingMeta:
        trialInfo = dataDF.index.to_frame().reset_index(drop=True)
        loadedTrialInfo = loadingMeta['listOfExampleIndexes'][segIdx].to_frame().reset_index(drop=True)
        targetAnns = np.intersect1d(trialInfo.columns, loadedTrialInfo.columns)
        targetAnns = [
            ta
            for ta in targetAnns
            if ta not in ['stimDelay', 'unitAnnotations', 'detectionDelay', 'originalIndex']]
        try:
            assert (trialInfo.loc[:, targetAnns] == loadedTrialInfo.loc[:, targetAnns]).all(axis=None)
        except Exception:
            traceback.print_exc()
            pdb.set_trace()
    listOfDataFrames.append(dataDF)
if arguments['verbose']:
    prf.print_memory_usage('Done loading')
if arguments['lazy']:
    dataReader.file.close()

exportDF = pd.concat(listOfDataFrames)
outputDFPath = os.path.join(
    dataFramesFolder,
    '{}_{}_{}_df{}.h5'.format(
        blockBaseName,
        arguments['window'],
        arguments['alignQuery'],
        iteratorSuffix))
if arguments['verbose']:
    prf.print_memory_usage('Saving {}'.format(outputDFPath))
exportDF.to_hdf(outputDFPath, arguments['selectionName'], mode='r+')
