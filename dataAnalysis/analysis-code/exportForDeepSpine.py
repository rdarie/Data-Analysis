"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('PS')   # generate postscript output by default
import seaborn as sns

from namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from scipy import stats
import re
import dataAnalysis.plotting.aligned_signal_plots as asp
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#
sns.set()
sns.set_color_codes("dark")
sns.set_context("talk")
sns.set_style("whitegrid")
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
alignSubFolder = os.path.join(
    analysisSubFolder, arguments['alignFolderName']
    )
#
alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
    namedQueries, **arguments)
alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
    ash.processUnitQueryArgs(
        namedQueries, analysisSubFolder, **arguments))
alignedAsigsKWargs.update(dict(
    duplicateControlsByProgram=False,
    makeControlProgram=False,
    metaDataToCategories=False,
    removeFuzzyName=False,
    decimate=5,
    windowSize=(-250e-3, 750e-3),
    transposeToColumns='feature', concatOn='columns',))
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
triggeredPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['inputBlockName'], arguments['window']))
outputPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_export.h5'.format(
        arguments['inputBlockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataReader, dataBlock = ns5.blockFromPath(
    triggeredPath, lazy=arguments['lazy'])
asigWide = ns5.alignedAsigsToDF(
    dataBlock, **alignedAsigsKWargs)
metaData = asigWide.index.to_frame()
elecNames = metaData['electrode'].unique()
elecRegex = r'\-([\S\s]*)\+([\S\s]*)'
elecChanNames = []
for comboName in elecNames:
    matches = re.search(elecRegex, comboName)
    if matches:
        cathodeName = matches.groups()[0]
        if cathodeName not in elecChanNames:
            elecChanNames.append(cathodeName)
        anodeName = matches.groups()[1]
        if anodeName not in elecChanNames:
            elecChanNames.append(anodeName)
eesColumns = pd.MultiIndex.from_tuples(
    [(eCN, 'amplitude') for eCN in sorted(elecChanNames)],
    names=['object', 'property']
    )
#
trialIndex = pd.Index(np.unique(metaData['bin']))
trialColumns = pd.MultiIndex.from_tuples(
    [
        ('hip_flexion_r', 'angle'), ('knee_angle_r', 'angle'),
        ('hip_flexion_l', 'angle'), ('knee_angle_l', 'angle'),
    ], names=['object', 'property'])
#
manualPeriod = 0.01
manualStimTimes = np.arange(0, 0.3 + manualPeriod, manualPeriod)
manualEESWaveform = trialIndex.isin(manualStimTimes)
# print(metaData.reset_index(drop=True))
# print(metaData['electrode'])
nullKinematics = pd.DataFrame(
    0, index=trialIndex, columns=trialColumns)
kinKey = '/sling/kinematics'
with pd.HDFStore(outputPath) as store:
    nullKinematics.to_hdf(store, kinKey)
eesIdx = 0
pdb.set_trace()
for stimName, stimGroup in asigWide.groupby(['electrode', 'RateInHz', 'nominalCurrent']):
    if stimGroup.groupby(['segment', 't']).ngroups < 10:
        continue
    matches = re.search(elecRegex, stimName[0])
    cathodeName = matches.groups()[0]
    anodeName = matches.groups()[1]
    for trialIdx, (trialName, trialGroup) in enumerate(stimGroup.groupby(['segment', 't'])):
        stimKey = '/sling/sheep/spindle_0/biophysical/ees_{:0>3}/stim'.format(eesIdx)
        #
        eesIdx += 1
        theseResults = pd.DataFrame(0, index=trialIndex, columns=eesColumns)
        theseResults.loc[:, (cathodeName, 'amplitude')] = manualEESWaveform * stimName[2]
        theseResults.loc[:, (anodeName, 'amplitude')] = manualEESWaveform * stimName[2] * (-1)
        for cName, lag in trialGroup.columns:
            if 'EmgEnv' in cName:
                mName = cName.split('EmgEnv')[0]
                theseResults.loc[:, (mName, 'EMG')] = trialGroup[cName].to_numpy()
            if ('caudal' in cName) or ('rostral' in cName):
                lfpName = cName[:-4]
                theseResults.loc[:, (lfpName, 'lfp')] = trialGroup[cName].to_numpy()
        with pd.HDFStore(outputPath) as store:
            theseResults.to_hdf(store, stimKey)
            store.get_storer(stimKey).attrs.metadata = {
                'globalIdx': eesIdx, 'combinationIdx': trialIdx,
                'electrode': stimName[0], 'RateInHz': stimName[1],
                'amplitude': stimName[2]}

if arguments['lazy']:
    dataReader.file.close()
