"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --exp=exp                              which experimental day to analyze
    --processAll                           process entire experimental day? [default: False]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: (pedalMovementCat==\'midPeak\')]
    --window=window                        process with short window? [default: long]
    --estimatorName=estimatorName          filename for resulting estimator [default: pca]
    --inputBlockName=inputBlockName        filename for resulting estimator [default: fr_sqrt]
    --unitQuery=unitQuery                  how to restrict channels? [default: (chanName.str.endswith(\'fr_sqrt#0\'))]
"""

import os
import dataAnalysis.helperFunctions.profiling as prf
#  import numpy as np
#  import pandas as pd
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.pipeline import make_pipeline, Pipeline
import joblib as jb
import pickle
from currentExperiment_alt import parseAnalysisOptions
from docopt import docopt
import gc
arguments = docopt(__doc__)
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['--trialIdx']), arguments['--exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['--processAll']:
    triggeredPath = os.path.join(
        scratchFolder,
        experimentName + '_trig_{}_{}.nix'.format(
            arguments['--inputBlockName'], arguments['--window']))
    estimatorPath = os.path.join(
        scratchFolder,
        experimentName + '_' + arguments['--estimatorName'] + '.joblib')
else:
    triggeredPath = os.path.join(
        scratchFolder,
        ns5FileName + '_trig_{}_{}.nix'.format(
            arguments['--inputBlockName'], arguments['--window']))
    estimatorPath = os.path.join(
        scratchFolder,
        ns5FileName + '_' + arguments['--estimatorName'] + '.joblib')

prf.print_memory_usage('before load data')
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    amplitudeColumn='amplitudeFuzzy',
    programColumn='programFuzzy',
    electrodeColumn='electrodeFuzzy',
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False)

dataQuery = '&'.join([
    #  '((RateInHz==100)|(RateInHz==0))',
    arguments['--alignQuery']
    ])

unitNames = None
nSeg = len(dataBlock.segments)
nComp = 50
estimator = IncrementalPCA(
    n_components=nComp,
    batch_size=int(3 * nComp))

for segIdx in range(nSeg):
    alignedAsigsDF = ns5.alignedAsigsToDF(
        dataBlock, unitNames,
        unitQuery=arguments['--unitQuery'], dataQuery=dataQuery, verbose=True,
        setIndex=False, getMetaData=False, decimate=2, whichSegments=[segIdx],
        **alignedAsigsKWargs)
    prf.print_memory_usage('just loaded firing rates, fitting')
    estimator.partial_fit(alignedAsigsDF.to_numpy())
    saveColumns = alignedAsigsDF.columns.to_list()
    del alignedAsigsDF
    gc.collect()

dataReader.file.close()
jb.dump(estimator, estimatorPath)

estimatorMetadata = {
    'trainingDataPath': os.path.basename(triggeredPath),
    'path': os.path.basename(estimatorPath),
    'name': arguments['--estimatorName'],
    'inputBlockName': arguments['--inputBlockName'],
    'inputFeatures': saveColumns,
    'dataQuery': dataQuery,
    'alignedAsigsKWargs': alignedAsigsKWargs
    }

with open(estimatorPath.replace('.joblib', '_meta.pickle'), 'wb') as f:
    pickle.dump(
        estimatorMetadata, f)
