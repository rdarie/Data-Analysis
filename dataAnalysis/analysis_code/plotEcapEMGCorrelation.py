"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisNameEMG=analysisNameEMG      append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --emgBlockSuffix=emgBlockSuffix        which trig_ block to pull [default: emg]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: stimOn]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output
matplotlib.use('Qt5Agg')   # generate interactive output
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import seaborn as sns
from dataAnalysis.analysis_code.namedQueries import namedQueries
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.preproc.ns5 as ns5
import os
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'blockIdx': '3', 'analysisNameEMG': 'loRes', 'emgBlockSuffix': 'emg', 'processAll': False,
        'unitQuery': 'isiemgenv', 'lazy': False, 'alignQuery': 'stimOn', 'verbose': False, 'window': 'XXS',
        'exp': 'exp202012171300', 'alignFolderName': 'stim'
        }
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
sns.set(
    context='notebook', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=0.5, color_codes=True)
#
analysisSubFolderEMG = os.path.join(
    scratchFolder, arguments['analysisNameEMG']
    )
alignSubFolder = os.path.join(
    analysisSubFolderEMG, arguments['alignFolderName']
    )
calcSubFolderEMG = os.path.join(alignSubFolder, 'dataframes')
figureOutputFolder = os.path.join(
    figureFolder, arguments['analysisNameEMG'])
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
#
resultPathEMG = os.path.join(
    calcSubFolderEMG,
    prefix + '_{}_{}_rauc.h5'.format(
        arguments['emgBlockSuffix'], arguments['window']))
#  Overrides
limitPages = None
amplitudeFieldName = 'nominalCurrent'
# amplitudeFieldName = 'amplitudeCat'
#  End Overrides
# ecapPath = os.path.join(nspFolder, 'Block0003_Config1_PNP_Info_Single_Trial20210422-194157.csv')
listOfEcapPaths = [
    'Block0003_Config1_PNP_Info_20210330-215808.csv',
    'Block0003_Config2_PNP_Info_20210401-182713.csv'
    ]
lOfEcapDFs = []
for ecapFileName in listOfEcapPaths:
    ecapDF = pd.read_csv(os.path.join(nspFolder, ecapFileName))
    lOfEcapDFs.append(ecapDF)
ecapDF = pd.concat(lOfEcapDFs)
ecapDF.columns = [cN.strip() for cN in ecapDF.columns]
#
mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
mapDF.loc[:, 'channelRepetition'] = mapDF['label'].apply(lambda x: x.split('_')[-1])
mapDF.loc[:, 'topoName'] = mapDF['label'].apply(lambda x: x[:-2])
mapDF.loc[:, 'whichArray'] = mapDF['elecName'].apply(lambda x: x[:-1])
mapDF.loc[:, 'elecID'] = mapDF['elecID'].astype(int)
mapDF.loc[mapDF['nevID'] > 64, 'nevID'] = mapDF.loc[mapDF['nevID'] > 64, 'nevID'] - 96 # fix
mapDF.loc[:, 'nevID'] = mapDF['nevID'].astype(int)
#
nevIDToLabelMap = mapDF.loc[:, ['nevID', 'topoName']].set_index('nevID')['topoName']
ecapDF.loc[:, 'electrode'] = ecapDF['stimElec'].map(nevIDToLabelMap)
recordingArray = 'rostral'
mapMask = (mapDF['whichArray'] == recordingArray) & (mapDF['channelRepetition'] == 'a')
elecIDToLabelMap = mapDF.loc[mapMask, ['elecID', 'label']].set_index('elecID')['label']
ecapDF.loc[:, 'recElec'] = ecapDF['recElec'].astype(int)
ecapDF.loc[:, 'feature'] = ecapDF['recElec'].map(nevIDToLabelMap)
ecapDF.rename(columns={'Amp': 'nominalCurrent', 'Freq': 'RateInHz'}, inplace=True)
ecapDF.loc[:, 'nominalCurrent'] = ecapDF['nominalCurrent'] * (-1)
annotNames = ['xcoords', 'ycoords', 'whichArray']
for annotName in annotNames:
    lookupSource = mapDF.loc[:, [annotName, 'label']].set_index('label')[annotName]
    ecapDF.loc[:, annotName] = ecapDF['feature'].map(lookupSource)
ecapMeasureNames = ['P1_y', 'N1_y', 'P2_y']
if 'Repeat' not in ecapDF.columns:
    ecapDF.loc[:, 'Repeat'] = 1
indexNames = ['electrode', 'nominalCurrent', 'RateInHz', 'Repeat', 'feature']
ecapWideDF = ecapDF.set_index(indexNames).loc[:, ecapMeasureNames]
ecapWideDF.columns.name = 'measurement'
ecapWideDF = ecapWideDF.unstack(level='feature')
#
print('loading {}'.format(resultPathEMG))
recCurve = pd.read_hdf(resultPathEMG, 'meanRAUC')
plotOpts = pd.read_hdf(resultPathEMG, 'meanRAUC_plotOpts')
emgPalette = plotOpts.loc[:, ['featureName', 'color']].set_index('featureName')['color']
rates = recCurve.index.get_level_values('RateInHz')
dbIndexMask = (rates < 30)
recCurveWideDF = recCurve.loc[dbIndexMask, :].groupby(ecapDF.index.names).mean()['rauc']
recCurveWideDF = recCurveWideDF.unstack(level='feature')
