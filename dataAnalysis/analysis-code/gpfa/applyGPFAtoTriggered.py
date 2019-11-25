"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: long]
    --alignQuery=alignQuery                choose a data subset? [default: midPeak]
    --inputBlockName=inputBlockName        filename for inputs [default: raster]
    --selector=selector                    filename if using a unit selector
    --unitQuery=unitQuery                  how to restrict channels?
"""

import dataAnalysis.helperFunctions.profiling as prf
import dataAnalysis.neuralTrajInterface.neural_traj_interface as nti
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import os
import pandas as pd
import numpy as np
import pdb
import dataAnalysis.preproc.ns5 as ns5
import joblib as jb
import dill as pickle
import subprocess
import h5py

from currentExperiment import parseAnalysisOptions
from docopt import docopt
from namedQueries import namedQueries
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = ash.processUnitQueryArgs(
    namedQueries, scratchFolder, **arguments)
alignedAsigsKWargs['verbose'] = arguments['verbose']

alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    removeFuzzyName=False, getMetaData=True,
    procFun=lambda wfdf: wfdf > 0,
    transposeToColumns='bin', concatOn='index'))

if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}.nix'.format(
        arguments['window']))
intermediatePath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}_for_gpfa_{}.mat'.format(
        arguments['window'], arguments['alignQuery']))
outputPath = os.path.join(
    scratchFolder,
    prefix + '_raster_{}_from_gpfa_{}.mat'.format(
        arguments['window'], arguments['alignQuery']))
modelName = '{}_{}_{}'.format(prefix, arguments['window'], arguments['alignQuery'])
modelPath = os.path.join(
    scratchFolder, 'gpfa_results',
    modelName,
    'gpfa_xDim{:0>2}'.format(gpfaOpts['xDim']) + '.mat'
    )
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath,
    lazy=arguments['lazy'])
alignedRastersDF = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)

if not os.path.exists(intermediatePath):
    nti.saveRasterForNeuralTraj(alignedRastersDF, intermediatePath)

if not os.path.exists(outputPath):
    # dataPath, modelPath, outputPath, baseDir
    gpfaArg = ', '.join([
        '\'' + intermediatePath + '\'',
        '\'' + modelPath + '\'',
        '\'' + outputPath + '\'',
        '\'' + scratchFolder + '\'',
        ])
    execStr = 'matlab -r \"extract_gpfa({}); exit\"'.format(gpfaArg)
    print(execStr)
    result = subprocess.run([execStr], shell=True)

#  with h5py.File(outputPath, 'r') as f:
from neo.io.proxyobjects import SpikeTrainProxy
import quantities as pq
f = h5py.File(outputPath, 'r')
alignedFactors = []
featureNames = ['gpfa{:0>3}'.format(i)for i in range(gpfaOpts['xDim'])]

exSt = dataBlock.filter(objects=SpikeTrainProxy)[0]
winSize = rasterOpts['windowSizes'][arguments['window']]
nBins = int((winSize[1] - winSize[0]) * exSt.sampling_rate.magnitude /gpfaOpts['binWidth'] - 1)
bins = (np.arange(nBins) * gpfaOpts['binWidth'] + gpfaOpts['binWidth'] / 2 ) / 1e3 + winSize[0]

for tIdx, grp in enumerate(f['seqNew']['xsm']):
    name = h5py.h5r.get_name(grp[0], f.id)
    dataArray = f[name].value[:nBins, :]
    thisDF = pd.DataFrame(dataArray, index=bins, columns=featureNames)
    thisDF.index.name = 'bin'
    thisDF.columns.name = 'feature'
    alignedFactors.append(thisDF)
factorsDF = pd.concat(alignedFactors, axis=0)
del alignedFactors

alignedRastersDF = alignedRastersDF.iloc[:, :nBins].stack().unstack('feature')
alignedRastersDF.index = alignedRastersDF.index.droplevel('bin')
factorsDF.set_index(alignedRastersDF.index, append=True, inplace=True)

masterBlock = ns5.alignedAsigDFtoSpikeTrain(factorsDF, dataBlock)

if arguments['lazy']:
    dataReader.file.close()
masterBlock = ns5.purgeNixAnn(masterBlock)
writer = ns5.NixIO(filename=triggeredPath.replace('raster', 'gpfa'))
writer.write_block(masterBlock, use_obj_names=True)
writer.close()