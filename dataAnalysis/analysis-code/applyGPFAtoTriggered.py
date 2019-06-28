"""  13: calc PCA of neural state aligned to an event
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --exp=exp                              which experimental day to analyze
    --verbose                              print diagnostics? [default: False]
    --alignQuery=alignQuery                choose a subset of the data?
    --modelSuffix=modelSuffix              what name to append in order to identify the gpfa model? [default: midPeak]
    --selector=selector                    filename if using a unit selector
    --window=window                        process with short window? [default: short]
    --unitQuery=unitQuery                  how to restrict channels? [default: (chanName.str.endswith(\'raster#0\'))]
"""
#  import dataAnalysis.plotting.aligned_signal_plots as asp
#  import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.helperFunctions.profiling as prf
import os
#  import seaborn as sns
#  import numpy as np
#  import quantities as pq
import pandas as pd
import numpy as np
import scipy.io as sio
import pdb
import dataAnalysis.preproc.ns5 as ns5
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  from neo.io.proxyobjects import (
#      AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import joblib as jb
import dill as pickle
import subprocess
import h5py
#  import gc

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_trig_raster_{}.nix'.format(
        arguments['window']))

modelPath = os.path.join(
    scratchFolder, 'gpfa_results',
    '{}_{}'.format(prefix, arguments['modelSuffix']),
    'gpfa_xDim{:0>2}'.format(gpfaOpts['xDim']) + '.mat'
    )
intermediatePath = triggeredPath.replace('.nix', '_for_gpfa_temp.mat')
outputPath = triggeredPath.replace('.nix', '_from_gpfa_temp.mat')

if arguments['verbose']:
    print('Loading {}...'.format(triggeredPath))
    prf.print_memory_usage('before load data')
dataReader = ns5.nixio_fr.NixIO(
    filename=triggeredPath)
dataBlock = dataReader.read_block(
    block_index=0, lazy=True,
    signal_group_mode='split-all')

if arguments['alignQuery'] is None:
    dataQuery = None
elif len(arguments['alignQuery']) == 0:
    dataQuery = None
else:
    dataQuery = '&'.join([
        arguments['alignQuery']
    ])

if arguments['verbose']:
    prf.print_memory_usage('before load firing rates')

if arguments['selector'] is not None:
    with open(
        os.path.join(
            scratchFolder,
            arguments['selector'] + '.pickle'),
            'rb') as f:
        selectorMetadata = pickle.load(f)
    unitNames = selectorMetadata['outputFeatures']
    alignedAsigsKWargs = selectorMetadata['alignedAsigsKWargs']
else:
    unitNames = None
    alignedAsigsKWargs = dict(
        duplicateControlsByProgram=False,
        makeControlProgram=True,
        removeFuzzyName=False)

if miniRCTrial:
    alignedAsigsKWargs.update(dict(
        amplitudeColumn='amplitude',
        programColumn='program',
        electrodeColumn='electrode'))

alignedRastersDF = ns5.alignedAsigsToDF(
    dataBlock, unitNames,
    unitQuery=arguments['unitQuery'], dataQuery=dataQuery,
    procFun=lambda wfdf: wfdf > 0,
    transposeToColumns='bin', concatOn='index',
    getMetaData=True,
    **alignedAsigsKWargs, verbose=arguments['verbose'])

#  keepMetaCols = ['segment', 'originalIndex', 'feature']
#  dropMetaCols = np.setdiff1d(alignedRastersDF.index.names, keepMetaCols).tolist()
#  alignedRastersDF.index = alignedRastersDF.index.droplevel(dropMetaCols)

if arguments['verbose']:
    prf.print_memory_usage('after load firing rates')

intermediateMatAlreadyExists = False
if not intermediateMatAlreadyExists:
    alignedRasterList = [
        g.to_numpy(dtype='uint8')
        for n, g in alignedRastersDF.groupby(['segment', 'originalIndex'])]
    trialIDs = [
        np.atleast_2d(i).astype('uint16')
        for i in range(len(alignedRasterList))]
    structDType = np.dtype([('trialId', 'O'), ('spikes', 'O')])

    dat = np.array(list(zip(trialIDs, alignedRasterList)), dtype=structDType)
    sio.savemat(intermediatePath, {'dat': dat})

outputMatAlreadyExists = False
if not outputMatAlreadyExists:
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
dataReader.file.close()
#  print('memory usage: {:.1f} MB'.format(prf.memory_usage_psutil()))
masterBlock = ns5.purgeNixAnn(masterBlock)
writer = ns5.NixIO(filename=triggeredPath.replace('raster', 'gpfa'))
writer.write_block(masterBlock, use_obj_names=True)
writer.close()