"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --noStim                               remove stim pulses? [default: False]
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockName=inputBlockName        which trig_ block to pull [default: pca]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                    filename if using a unit selector
"""
import matplotlib
# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42
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

@profile
def exportForDeepSpineWrapper():
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
    calcSubFolder = os.path.join(alignSubFolder, 'dataframes')
    if not os.path.exists(calcSubFolder):
        os.makedirs(calcSubFolder, exist_ok=True)

    if arguments['processAll']:
        prefix = assembledName
    else:
        prefix = ns5FileName
    alignedAsigsKWargs['dataQuery'] = ash.processAlignQueryArgs(
        namedQueries, **arguments)
    alignedAsigsKWargs['unitNames'], alignedAsigsKWargs['unitQuery'] = (
        ash.processUnitQueryArgs(
            namedQueries, analysisSubFolder, **arguments))
    outlierTrialNames = ash.processOutlierTrials(
        calcSubFolder, prefix, **arguments)

    if arguments['window'] == 'XS':
        cropWindow = (-100e-3, 400e-3)
    elif arguments['window'] == 'XSPre':
        cropWindow = (-600e-3, -100e-3)

    alignedAsigsKWargs.update(dict(
        duplicateControlsByProgram=False,
        makeControlProgram=False,
        metaDataToCategories=False,
        removeFuzzyName=False,
        decimate=1,
        windowSize=cropWindow,
        transposeToColumns='feature', concatOn='columns',))
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
    # asigWide is a dataframe
    metaData = asigWide.index.to_frame()
    elecNames = metaData['electrode'].unique()

    # elecRegex = r'([\-]?[\S\s]*\d)([\+]?[\S\s]*\d)'
    # elecRegex = r'((?:\-|\+)(?:(?:rostral|caudal)\S_\S\S\S)*)*'

    elecRegex = r'((?:\-|\+)(?:(?:rostral|caudal)\S_\S\S\S)*)'
    chanRegex = r'((?:rostral|caudal)\S_\S\S\S)'
    elecChanNames = []
    stimConfigLookup = {}
    for comboName in elecNames:
        matches = re.findall(elecRegex, comboName)
        if matches:
            print(comboName)
            thisLookup = {'cathodes': [], 'anodes': []}
            for matchGroup in matches:
                print('\t' + matchGroup)
                if len(matchGroup):
                    theseChanNames = re.findall(chanRegex, matchGroup)
                    if theseChanNames:
                        for chanName in theseChanNames:
                            if chanName not in elecChanNames:
                                elecChanNames.append(chanName)
                        if '-' in matchGroup:
                            for chanName in theseChanNames:
                                if chanName not in thisLookup['cathodes']:
                                    thisLookup['cathodes'].append(chanName)
                        if '+' in matchGroup:
                            for chanName in theseChanNames:
                                if chanName not in thisLookup['anodes']:
                                    thisLookup['anodes'].append(chanName)
            stimConfigLookup[comboName] = thisLookup

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
    # manualPeriod = 0.01
    # manualStimTimes = np.arange(0, 0.3 + manualPeriod, manualPeriod)
    # manualEESWaveform = trialIndex.isin(manualStimTimes)
    # print(metaData.reset_index(drop=True))
    # print(metaData['electrode'])
    nullKinematics = pd.DataFrame(
        0, index=trialIndex, columns=trialColumns)
    kinKey = '/sling/kinematics'
    with pd.HDFStore(outputPath) as store:
        nullKinematics.to_hdf(store, kinKey)
    eesIdx = 0

    for stimName, stimGroup in asigWide.groupby(['electrode', 'RateInHz', 'nominalCurrent']):
        if stimGroup.groupby(['segment', 't']).ngroups < 5:
            continue
        print(stimName)
        for trialIdx, (trialName, trialGroup) in enumerate(stimGroup.groupby(['segment', 't'])):
            stimKey = '/sling/sheep/spindle_0/biophysical/ees_{:0>3}/stim'.format(eesIdx)
            eesPeriod = stimName[1] ** -1
            stimTimes = np.arange(0, 0.3, eesPeriod)
            EESWaveform = np.zeros_like(trialIndex)
            # TODO replace this with the hf.findClosestTimes implementation
            if not arguments['noStim']:
                for stimTime in stimTimes:
                    closestIndexTime = np.argmin(np.abs((trialIndex - stimTime)))
                    EESWaveform[closestIndexTime] = 1
            eesIdx += 1
            theseResults = pd.DataFrame(0, index=trialIndex, columns=eesColumns)
            for cathodeName in stimConfigLookup[stimName[0]]['cathodes']:
                theseResults.loc[:, (cathodeName, 'amplitude')] = EESWaveform * stimName[2] / len(stimConfigLookup[stimName[0]]['cathodes'])
            for anodeName in stimConfigLookup[stimName[0]]['anodes']:
                theseResults.loc[:, (anodeName, 'amplitude')] = EESWaveform * stimName[2] * (-1) / len(stimConfigLookup[stimName[0]]['anodes'])
            for cName, lag in trialGroup.columns:
                if 'EmgEnv' in cName:
                    mName = cName.split('EmgEnv')[0]
                    theseResults.loc[:, (mName, 'emg_env')] = trialGroup[cName].to_numpy()
                elif 'Emg' in cName:
                    mName = cName.split('Emg')[0]
                    theseResults.loc[:, (mName, 'emg')] = trialGroup[cName].to_numpy()
                elif ('caudal' in cName) or ('rostral' in cName):
                    lfpName = cName[:-4]
                    theseResults.loc[:, (lfpName, 'lfp')] = trialGroup[cName].to_numpy()
                elif ('Acc' in cName):
                    nameParts = cName.split('Acc')
                    mName = nameParts[0]
                    theseResults.loc[:, (mName, 'acc_{}'.format(nameParts[1][0].lower()))] = trialGroup[cName].to_numpy()
            with pd.HDFStore(outputPath) as store:
                theseResults.to_hdf(store, stimKey)
                thisMetadata = {
                    'globalIdx': eesIdx, 'combinationIdx': trialIdx,
                    'electrode': stimName[0], 'RateInHz': stimName[1],
                    'amplitude': stimName[2]}
                if arguments['maskOutlierBlocks']:
                    thisMetadata['outlierTrial'] = outlierTrialNames.loc[trialName]
                store.get_storer(stimKey).attrs.metadata = thisMetadata

    if arguments['lazy']:
        dataReader.file.close()
    return

if __name__ == "__main__":
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=exportForDeepSpineWrapper,
            modulesToProfile=[ash, ns5],
            #outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        exportForDeepSpineWrapper()
