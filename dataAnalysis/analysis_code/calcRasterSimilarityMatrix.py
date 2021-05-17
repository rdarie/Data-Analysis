"""
Usage:
    calcSimilarityMatrix.py [options]

Options:
    --blockIdx=blockIdx                 which trial to analyze [default: 1]
    --exp=exp                           which experimental day to analyze [default: exp201901271000]
    --processAll                        process entire experimental day? [default: False]
    --lazy                              load from raw, or regular? [default: False]
    --window=window                     process with short window? [default: long]
    --analysisName=analysisName         append a name to the resulting blocks? [default: default]
    --unitQuery=unitQuery               how to select channels if not supplying a list? [default: neural]
    --alignQuery=alignQuery             query what the units will be aligned to? [default: midPeak]
    --selector=selector                 filename if using a unit selector
    --estimatorName=estimatorName       filename for resulting estimator [default: umap]
    --verbose                           print diagnostics? [default: False]
    --plotting                          plot out the correlation matrix? [default: True]
"""
#  The text block above is used by the docopt package to parse command line arguments
#  e.g. you can call <python3 calcBlockSimilarityMatrix.py> to run with default arguments
#  but you can also specify, for instance <python3 calcBlockSimilarityMatrix.py --blockIdx=2>
#  to load trial 002 instead of trial001 which is the default
#
#  regular package imports
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("dark")
import matplotlib.ticker as ticker
import os, pdb, traceback
from importlib import reload
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import elephant as elph
import umap
import dill as pickle
import joblib as jb
from sklearn.manifold import TSNE
#   you can specify options related to the analysis via the command line arguments,
#   or by saving variables in the currentExperiment.py file, or the individual exp2019xxxxxxxx.py files
#
#   these lines process the command line arguments
#   they produces a python dictionary called arguments
from namedQueries import namedQueries
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
#  this stuff imports variables from the currentExperiment.py and exp2019xxxxxxxx.py files
from currentExperiment import parseAnalysisOptions
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#   once these lines run, your workspace should contain a bunch of variables that were imported
#   you can check them by calling the python functions globals() for global variables and locals() for local variables
#   both of these functions return python dictionaries
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_raster_{}.nix'.format(arguments['window']))
fullEstimatorName = '{}_{}_{}_{}'.format(
    prefix,
    arguments['estimatorName'],
    arguments['window'],
    arguments['alignQuery'])
estimatorPath = os.path.join(
    analysisSubFolder,
    fullEstimatorName + '.joblib')
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, analysisSubFolder, inputBlockName='raster', **arguments)
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    transposeToColumns='bin', concatOn='index',
    removeFuzzyName=False,
    getMetaData=[
        'RateInHz', 'activeGroup', 'amplitude', 'amplitudeCat',
        'bin', 'electrode', 'pedalDirection', 'pedalMetaCat',
        'pedalMovementCat', 'pedalMovementDuration',
        'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
        'program', 'segment', 't'],
    decimate=1,
    metaDataToCategories=False,
    verbose=False, procFun=None))
#
reloadFeatures = True
reloadEstimator = True
ssimsOpts = {
    'distanceOpts':
    {
        'maxUnit': None,
        'distanceFun': elph.spike_train_dissimilarity.victor_purpura_dist,
        'distanceFunKWargs': {
            'q': (30 * pq.ms) ** (-1)
            }
        },
    'reducerOpts':
    {
        'reducerClass': umap.UMAP,
        'reducerKWargs': {
            'n_neighbors': 20, 'min_dist': 0.1,
            'n_components': 2, 'metric': 'euclidean'}
            }
    #  {
    #      'reducerClass': TSNE,
    #      'reducerKWargs': {
    #          'n_components': 2}
    #          }
    }
similarityMetaDataPath = triggeredPath.replace(
    '.nix', '_similarity_meta.pickle')
similarityH5Path = triggeredPath.replace(
    '.nix', '_similarity.h5')
#
if os.path.exists(similarityMetaDataPath):
    with open(similarityMetaDataPath, 'rb') as f:
        similarityMetaData = pickle.load(f)
    sameFeatures = (similarityMetaData['alignedAsigsKWargs'] == alignedAsigsKWargs)
    sameDistOpts = (similarityMetaData['ssimsOpts']['distanceOpts'] == ssimsOpts['distanceOpts'])
    sameRedOpts = (similarityMetaData['ssimsOpts']['reducerOpts'] == ssimsOpts['reducerOpts'])
    if sameFeatures and sameDistOpts:
        print('Reusing distance matrix')
        reloadFeatures = False
        simDF = pd.read_hdf(similarityH5Path, 'similarity')
    if sameRedOpts:
        print('Reusing estimator')
        reloadEstimator = False
        estimator = jb.load(estimatorPath)
#
if reloadFeatures or reloadEstimator:
    globals().update(ssimsOpts)

if reloadFeatures:
    if arguments['verbose']:
        print('Loading dataBlock: {}'.format(triggeredPath))
    dataReader, dataBlock = ns5.blockFromPath(
        triggeredPath, lazy=arguments['lazy'])
    if arguments['verbose']:
        print('Loading alignedAsigs: {}'.format(triggeredPath))
    rasterDF = ns5.alignedAsigsToDF(
        dataBlock, **alignedAsigsKWargs)
    #
    idxNames = rasterDF.index.names
    windowSize = [rasterDF.columns.min() * pq.s, rasterDF.columns.max() * pq.s]
    #
    def recoverSpikeTrain(Ser):
        times = Ser.index[Ser > 0].to_numpy(dtype=float) * pq.s
        anns = {idxNames[i]: ann for i, ann in enumerate(Ser.name)}
        t_start, t_stop = windowSize
        return SpikeTrain(times, t_stop=t_stop, t_start=t_start, **anns)
    #
    allSimMat = []
    for uIdx, (name, group) in enumerate(rasterDF.groupby('feature')):
        if uIdx == 0:
            metaData = group.index
        if arguments['verbose']:
            print('Calculating distance for unit {}'.format(uIdx))
        unitSpikeTrains = [recoverSpikeTrain(tr) for meta, tr in group.iterrows()]
        unitSimMat = distanceOpts['distanceFun'](
            unitSpikeTrains, **distanceOpts['distanceFunKWargs'])
        allSimMat.append(unitSimMat)
        if distanceOpts['maxUnit'] is not None:
            if uIdx >= distanceOpts['maxUnit']:
                break
    #
    simDF = pd.DataFrame(
        np.concatenate(allSimMat, axis=1),
        index=metaData,
        columns=pd.MultiIndex.from_product(
            [
                rasterDF.index.get_level_values('feature').unique()[:uIdx + 1],
                list(range(metaData.shape[0]))
            ], names=['feature', 'trial'])
        )
    #
    simDF.to_hdf(similarityH5Path, 'similarity')
#
if reloadEstimator:
    estimator = reducerOpts['reducerClass'](**reducerOpts['reducerKWargs'])
    estimator.fit(simDF)
    jb.dump(estimator, estimatorPath)

similarityMetaData = {
    'alignedAsigsKWargs': alignedAsigsKWargs,
    'ssimsOpts': ssimsOpts
}
with open(similarityMetaDataPath, 'wb') as f:
    pickle.dump(similarityMetaData, f)

if 'embedding_' in dir(estimator):
    embedding_ = estimator.embedding_
else:
    embedding_ = estimator.fit_transform(simDF.to_numpy())
#
trainingQuery = "(pedalSizeCat == 'M')"
alignedAsigsKWargs['getMetaData'] = [
    'RateInHzFuzzy', 'activeGroup', 'amplitudeFuzzy', 'amplitudeFuzzyCat',
    'bin', 'electrodeFuzzy', 'pedalDirection', 'pedalMetaCat',
    'pedalMovementCat', 'pedalMovementDuration',
    'pedalSize', 'pedalSizeCat', 'pedalVelocityCat',
    'programFuzzy', 'segment', 't']
if trainingQuery is not None:
    trainDF = simDF.query(trainingQuery)
    embIndex = trainDF.index
else:
    trainDF = simDF
    embIndex = simDF.index
embedding_ = estimator.fit_transform(trainDF.to_numpy())
embedding = pd.DataFrame(
    embedding_,
    index=rasterDF.query(trainingQuery).index,
    #index=embIndex,
    columns=[
        '{}_{}'.format(arguments['estimatorName'], i)
        for i in range(embedding_.shape[1])])
embPlot = embedding.reset_index()
embPlot['ACRFuzzy'] = (
    embPlot['amplitudeFuzzy'] *
    embPlot['RateInHzFuzzy'] /
    1000)
pdfPath = os.path.join(
    figureFolder,
    prefix + '_{}_ssims.pdf'.format(
        arguments['estimatorName']))
cPalette = 'ch:0,0.4,dark=.2,light=0.6'
embPlot['amplitudeFuzzy'] = embPlot['amplitudeFuzzy'] / 1e3
embPlot['amplitudeFuzzy'] = embPlot['amplitudeFuzzy'].round(decimals = 1)
ax = sns.scatterplot(
    x='{}_0'.format(arguments['estimatorName']),
    y='{}_1'.format(arguments['estimatorName']),
    hue='amplitudeFuzzy',
    size='amplitudeFuzzy', sizes=(50, 200),
    style='electrodeFuzzy',
    data=(
        embPlot
        .query("(programFuzzy == 0) | (programFuzzy == 2)")
        ),
    palette=cPalette
    ); plt.show()
plt.savefig(pdfPath)
# plt.close()
plt.show()
