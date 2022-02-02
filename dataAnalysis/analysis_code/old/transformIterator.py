"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --exp=exp                                     which experimental day to analyze
    --blockIdx=blockIdx                           which trial to analyze [default: 1]
    --analysisName=analysisName                   append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName             append a name to the resulting blocks? [default: motion]
    --processAll                                  process entire experimental day? [default: False]
    --verbose=verbose                             print diagnostics? [default: 0]
    --lazy                                        load from raw, or regular? [default: False]
    --window=window                               process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix           which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix           which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                         how to restrict channels if not supplying a list? [default: pca]
    --alignQuery=alignQuery                       what will the plot be aligned to? [default: outboundWithStim]
    --selector=selector                           filename if using a unit selector
    --maskOutlierBlocks                           delete outlier trials? [default: False]
    --winStart=winStart                           start of absolute window (when loading)
    --winStop=winStop                             end of absolute window (when loading)
    --loadFromFrames                              delete outlier trials? [default: False]
    --plotting                                    delete outlier trials? [default: False]
    --showFigures                                 delete outlier trials? [default: False]
    --datasetName=datasetName                     filename for resulting estimator (cross-validated n_comps)
    --selectionName=selectionName                 filename for resulting estimator (cross-validated n_comps)
    --iteratorSuffix=iteratorSuffix                   filename for resulting estimator (cross-validated n_comps)
    --iteratorOutputName=iteratorOutputName       filename for resulting estimator (cross-validated n_comps)
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')   # generate postscript output by default
matplotlib.use('QT5Agg')   # generate interactive output
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.custom_transformers.tdr as tdr
import dataAnalysis.preproc.ns5 as ns5
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
import os
from dask.distributed import Client, LocalCluster
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer, PowerTransformer
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import sys
sns.set(
    context='paper', style='whitegrid',
    palette='dark', font='sans-serif',
    font_scale=1., color_codes=True, rc={
        'figure.dpi': 200, 'savefig.dpi': 200})
'''
consoleDebug = True
if consoleDebug:
    arguments = {
    'processAll': True, 'blockIdx': '2', 'selectionName': 'lfp_CAR_spectral_fa_mahal',
    'lazy': False, 'verbose': False, 'window': 'XL', 'alignQuery': 'starting', 'datasetName': 'Block_XL_df_f',
    'analysisName': 'default', 'exp': 'exp202101281100', 'selector': None, 'winStart': '200', 'alignFolderName': 'motion',
    'winStop': '400', 'loadFromFrames': True, 'maskOutlierBlocks': False, 'unitQuery': 'fr_sqrt', 'inputBlockPrefix': 'Block',
    'inputBlockSuffix': 'lfp_CAR_spectral_mahal', 'iteratorSuffix': 'f', 'plotting': True}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''
for arg in sys.argv:
    print(arg)
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#

if __name__ == "__main__":
    blockBaseName, inputBlockSuffix = hf.processBasicPaths(arguments)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName']
    )
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName']
    )
    calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
    if arguments['iteratorSuffix'] is not None:
        iteratorSuffix = '_{}'.format(arguments['iteratorSuffix'])
    else:
        iteratorSuffix = ''
    if arguments['iteratorOutputName'] is not None:
        iteratorOutputName = '_{}'.format(arguments['iteratorOutputName'])
    else:
        iteratorOutputName = ''
    if not arguments['loadFromFrames']:
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_rauc_iterator{}.pickle'.format(
                inputBlockSuffix, arguments['window'], iteratorSuffix))
        iteratorOutputPath = os.path.join(
            calcSubFolder,
            blockBaseName + '{}_{}_rauc_iterator{}.pickle'.format(
                inputBlockSuffix, arguments['window'], iteratorOutputName))
    else:
        # loading from dataframe
        selectionName = arguments['selectionName']
        iteratorPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc_iterator{}.pickle'.format(
                selectionName, arguments['window'], iteratorSuffix))
        iteratorOutputPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc_iterator{}.pickle'.format(
                selectionName, arguments['window'], iteratorOutputName))
    with open(iteratorPath, 'rb') as _f:
        iteratorMeta = pickle.load(_f)
        iteratorKWArgs = iteratorMeta['iteratorKWArgs']
        cvIterator = iteratorMeta['cvIterator']

    if arguments['showFigures']:
        # if True:
        fig, ax = cvIterator.plot_schema()
        fig.suptitle('Before')
        plt.show()
    transformType = 'removeResampler'
    if transformType == 'removeResampler':
        iteratorKWArgs['resamplerClass'] = None
        iteratorKWArgs['resamplerKWArgs'] = {}
        cvIterator.splitter.resampler = None
        cvIterator.prelimSplitter.resampler = None
        cvIterator.folds = cvIterator.raw_folds.copy()
        cvIterator.workIterator.folds = [(cvIterator.work, cvIterator.validation,)]

    if arguments['showFigures']:
        # if True:
        fig, ax = cvIterator.plot_schema()
        fig.suptitle('After')
        plt.show()

    iteratorInfo = {
        'iteratorKWArgs': iteratorKWArgs,
        'cvIterator': cvIterator
        }
    with open(iteratorOutputPath, 'wb') as _f:
        pickle.dump(iteratorInfo, _f)
