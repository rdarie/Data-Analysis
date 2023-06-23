"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --expList=expList                      which experimental day to analyze
    --selectionList=selectionList          which signal type to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --processAll                           process entire experimental day? [default: False]
    --verbose                              print diagnostics? [default: True]
    --lazy                                 load from raw, or regular? [default: False]
    --window=window                        process with short window? [default: short]
    --inputBlockSuffix=inputBlockSuffix    which trig_ block to pull [default: pca]
    --inputBlockPrefix=inputBlockPrefix    which trig_ block to pull [default: Block]
    --unitQuery=unitQuery                  how to restrict channels if not supplying a list? [default: pca]
    --plotSuffix=plotSuffix                switches between a few different processing options [default: all]
    --alignQuery=alignQuery                what will the plot be aligned to? [default: outboundWithStim]
    --freqBandGroup=freqBandGroup          what will the plot be aligned to? [default: outboundWithStim]
    --maskOutlierBlocks                    delete outlier trials? [default: False]
    --invertOutlierMask                    delete outlier trials? [default: False]
    --showFigures                          show plots interactively? [default: False]
    --plotThePieces                        show plots interactively? [default: False]
    --plotTheAverage                       show plots interactively? [default: False]
    --plotTopoEffectSize                  show plots interactively? [default: False]
"""
import logging, sys
logging.captureWarnings(True)
import os
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
if 'CCV_HEADLESS' in os.environ:
    mpl.use('Agg')   # generate postscript output
else:
    mpl.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
from dataAnalysis.analysis_code.namedQueries import namedQueries
from dataAnalysis.analysis_code.currentExperiment import *
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pdb, traceback
import dataAnalysis.plotting.aligned_signal_plots as asp
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.helper_functions_new as hf
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as ns5
import os
from itertools import product
from docopt import docopt
import dill as pickle
import pandas as pd
import numpy as np
import pingouin as pg
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from datetime import datetime as dt
import contextlib
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
useDPI = 300
dpiFactor = 72 / useDPI
snsContext = 'paper'
snsRCParams = {
    'axes.facecolor': 'w',
    #
    "xtick.direction": 'in',
    "ytick.direction": 'in',
    #
    "axes.spines.left": False,
    "axes.spines.bottom": True,
    "axes.spines.right": True,
    "axes.spines.top": False,
    #
    "xtick.bottom": True,
    "xtick.top": False,
    "ytick.left": True,
    "ytick.right": False,
    }
snsRCParams.update(customSeabornContexts[snsContext])
mplRCParams = {
    'figure.dpi': useDPI, 'savefig.dpi': useDPI,
    #
    'axes.titlepad': customSeabornContexts[snsContext]['font.size'] * 0.5 + 2.,
    'axes.labelpad': 0.75,
    #
    'figure.subplot.left': 0.02,
    'figure.subplot.right': 0.98,
    'figure.subplot.bottom': 0.02,
    'figure.subplot.top': 0.98,
    #
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    }

styleOpts = {
    'legend.lw': 2,
    'tight_layout.pad': 1.,
    'panel_heading.pad': 2.
    }

for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

sns.set(
    context=snsContext, style='white',
    palette='dark', font='sans-serif',
    color_codes=True, rc=snsRCParams)

arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}

consoleDebug = False
'''
consoleDebug = True
if consoleDebug:
    arguments = {
        'window': 'XL', 'unitQuery': 'pca', 'plotTopoEffectSize': False,
        'alignFolderName': 'motion', 'plotThePieces': False, 'plotTheAverage': False,
        'selectionList': 'laplace_spectral_scaled, laplace_scaled, laplace_spectral_scaled_mahal_ledoit, laplace_scaled_mahal_ledoit',
        'showFigures': False, 'invertOutlierMask': False, 'blockIdx': '2',
        'expList': 'exp202101281100, exp202102021100, exp202101221100, exp202101251100, exp202101271100, exp202101211100',
        'processAll': True, 'verbose': False, 'lazy': False, 'alignQuery': 'starting',
        'inputBlockPrefix': 'Block', 'analysisName': 'hiRes', 'inputBlockSuffix': 'pca',
        'maskOutlierBlocks': False}
    os.chdir('/gpfs/home/rdarie/nda2/Data-Analysis/dataAnalysis/analysis_code')
    '''

pdfNameSuffix = ''
if arguments['plotSuffix'] == 'all':
    pdfNameSuffix = 'RAUC_all'
    lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CW_return', 'CCW_outbound', 'CCW_return']
elif arguments['plotSuffix'] == 'rest_stim':
    pdfNameSuffix = 'RAUC_rest_stim'
    lOfKinematicsToPlot = ['NA_NA']
elif arguments['plotSuffix'] == 'outbound_stim':
    pdfNameSuffix = 'RAUC_outbound_stim'
    lOfKinematicsToPlot = ['CCW_outbound', 'CW_outbound']
elif arguments['plotSuffix'] == 'outbound_E04':
    pdfNameSuffix = 'RAUC_outbound_E04'
    lOfKinematicsToPlot = ['CCW_outbound', 'CW_outbound']
elif arguments['plotSuffix'] == 'return_E04':
    pdfNameSuffix = 'RAUC_return_E04'
    lOfKinematicsToPlot = ['CCW_return', 'CW_return']
elif arguments['plotSuffix'] == 'E04':
    pdfNameSuffix = 'RAUC_E04'
    lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CW_return']
elif arguments['plotSuffix'] == 'rest_E04':
    pdfNameSuffix = 'RAUC_rest_E04'
    lOfKinematicsToPlot = ['NA_NA']
elif arguments['plotSuffix'] == 'move_E09':
    pdfNameSuffix = 'RAUC_move_E09'
    lOfKinematicsToPlot = ['CW_outbound', 'CW_return', 'CCW_outbound', 'CCW_return']
elif arguments['plotSuffix'] == 'best_three':
    pdfNameSuffix = 'RAUC_best_three'
    lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CCW_outbound']
#

basePalette = pd.Series(sns.color_palette('Paired'))
allAmpPalette = pd.Series(
    basePalette.apply(sns.utils.alter_color, h=-0.05).to_numpy()[:4],
    index=[
        'trialAmplitude', 'trialAmplitude_md',
        'trialAmplitude:trialRateInHz', 'trialAmplitude:trialRateInHz_md'])
allRelPalette = pd.Series(
    basePalette.apply(sns.utils.alter_color, h=0.05).to_numpy()[:6],
    index=['50.0', '50.0_md', '100.0', '100.0_md', '0.0', '0.0_md', ])
# reorder consistently
allRelPalette = allRelPalette.loc[['0.0', '0.0_md', '50.0', '50.0_md', '100.0', '100.0_md', ]]
# allRelPalette.loc['0.0'] = sns.utils.alter_color(allRelPalette.loc['50.0'], l=0.5)
# allRelPalette.loc['0.0_md'] = sns.utils.alter_color(allRelPalette.loc['50.0_md'], l=0.5)

'''
if True:
    fig, ax = plt.subplots(3, 1)
    basePalette = pd.Series(sns.color_palette('Paired'))
    sns.palplot(basePalette.apply(sns.utils.alter_color, h=-0.05).to_numpy()[:6], ax=ax[0])
    sns.palplot(basePalette.to_numpy()[:6], ax=ax[1])
    sns.palplot(basePalette.apply(sns.utils.alter_color, h=0.05).to_numpy()[:6], ax=ax[2])
    plt.show()'''
blockBaseName = arguments['inputBlockPrefix']
listOfExpNames = [x.strip() for x in arguments['expList'].split(',')]
listOfSelectionNames = [x.strip() for x in arguments['selectionList'].split(',')]
recCurveList = []
ampStatsDict = {}
relativeStatsDict = {}
relativeStatsNoStimDict = {}
ampStatsPerFBDict = {}
relativeStatsPerFBDict = {}
compoundAnnLookupList = []
featureInfoList = []
for expIdx, expName in enumerate(listOfExpNames):
    arguments['exp'] = expName
    expOpts, allOpts = parseAnalysisOptions(
        int(arguments['blockIdx']), arguments['exp'])
    globals().update(expOpts)
    globals().update(allOpts)
    analysisSubFolder = os.path.join(
        scratchFolder, arguments['analysisName'])
    #
    alignSubFolder = os.path.join(
        analysisSubFolder, arguments['alignFolderName'])
    calcSubFolder = os.path.join(analysisSubFolder, 'dataframes')
    outlierTrials = ash.processOutlierTrials(
        scratchFolder, blockBaseName, **arguments)

    if expIdx == 0:
        spinalMapDF = spinalElectrodeMaps[subjectName].sort_values(['xCoords', 'yCoords'])
        if arguments['plotSuffix'] == 'all':
            lOfElectrodesToPlot = spinalMapDF.index.to_list()
        elif arguments['plotSuffix'] == 'rest_stim':
            lOfElectrodesToPlot = spinalMapDF.index.to_list()
        elif arguments['plotSuffix'] == 'outbound_stim':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E05+E16', '-E11+E16', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'move_E09':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E09+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'outbound_E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'return_E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'rest_E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'best_three':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E05+E16', '-E11+E16', '-E04+E16'] if eN in spinalMapDF.index]
    for inputBlockSuffix in listOfSelectionNames:
        resultPath = os.path.join(
            calcSubFolder,
            blockBaseName + '_{}_{}_rauc.h5'.format(
                inputBlockSuffix, arguments['window']))
        signalIsMahalDist = 'mahal' in inputBlockSuffix
        print('loading {}'.format(resultPath))
        if not os.path.exists(resultPath):
            print('WARNING: path does not exist\n{}'.format(resultPath))
            continue
        try:
            rawRecCurve = pd.read_hdf(resultPath, 'raw')
            thisRecCurveFeatureInfo = rawRecCurve.columns.to_frame().reset_index(drop=True)
            rawRecCurve.columns = rawRecCurve.columns.get_level_values('feature')
            thisRecCurve = rawRecCurve.stack().to_frame(name='rawRAUC')
            scaledRaucDF = pd.read_hdf(resultPath, 'boxcox')
            scaledRaucDF.columns = scaledRaucDF.columns.get_level_values('feature')
            thisRecCurve.loc[:, 'scaledRAUC'] = scaledRaucDF.stack().to_numpy()
            for iN in ['isOutlierTrial', 'outlierDeviation']:
                if iN in thisRecCurve.index.names:
                    thisRecCurve.index = thisRecCurve.index.droplevel(iN)
            # relativeRaucDF = pd.read_hdf(resultPath, 'relative')
            # relativeRaucDF.columns = relativeRaucDF.columns.get_level_values('feature')
            # thisRecCurve.loc[:, 'normalizedRAUC'] = relativeRaucDF.stack().to_numpy()
            # clippedRaucDF = pd.read_hdf(resultPath, 'clipped')
            # clippedRaucDF.columns = clippedRaucDF.columns.get_level_values('feature')
            # thisRecCurve.loc[:, 'clippedRAUC'] = clippedRaucDF.stack().to_numpy()
            #
            thisRecCurve.loc[:, 'freqBandName'] = thisRecCurve.index.get_level_values('feature').map(thisRecCurveFeatureInfo[['feature', 'freqBandName']].set_index('feature')['freqBandName'])
            thisRecCurve.set_index('freqBandName', append=True, inplace=True)
            ################################################################################################################
            ################################################################################################################
            recCurveTrialInfo = thisRecCurve.index.to_frame().reset_index(drop=True)
            recCurveTrialInfo.loc[:, 'kinematicCondition'] = recCurveTrialInfo['kinematicCondition'].apply(lambda x: x.replace('CCW_', 'CW_'))
            if 'recCurveTrialInfoColumnOrder' not in locals():
                recCurveTrialInfoColumnOrder = recCurveTrialInfo.columns.to_list()
            else:
                recCurveTrialInfo = recCurveTrialInfo.loc[:, recCurveTrialInfoColumnOrder]
            thisRecCurve.index = pd.MultiIndex.from_frame(recCurveTrialInfo)
            '''
            if (subjectName == 'Murdoc'):
                hackMask1 = recCurveTrialInfo['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']).any()
                if hackMask1:
                    dropMask = recCurveTrialInfo['electrode'].isin(['-E09+E16', '-E00+E16']).to_numpy()
                    thisRecCurve = thisRecCurve.loc[~dropMask, :]
                    recCurveTrialInfo = thisRecCurve.index.to_frame().reset_index(drop=True)
                hackMask2 = recCurveTrialInfo['expName'].isin(['exp201902031100']).any()
                if hackMask2:
                    dropMask = recCurveTrialInfo['electrode'].isin(['-E00+E16']).to_numpy()
                    thisRecCurve = thisRecCurve.loc[~dropMask, :]
                    recCurveTrialInfo = thisRecCurve.index.to_frame().reset_index(drop=True)
                    '''
            ################################################################################################################
            ################################################################################################################
            thisRecCurve.loc[:, 'isMahalDist'] = signalIsMahalDist
            thisRecCurveFeatureInfo.loc[:, 'isMahalDist'] = signalIsMahalDist
            recCurveList.append(thisRecCurve)
            # print('thisRecCurve.index.names =\n{}'.format(',  '.join(thisRecCurve.index.names)))
            print('recCurveTrialInfo.T =\n{}'.format(recCurveTrialInfo.T))
            #####
            try:
                thisRecCurveFeatureInfo.loc[:, 'xIdx'], thisRecCurveFeatureInfo.loc[:, 'yIdx'] = ssplt.coordsToIndices(
                    thisRecCurveFeatureInfo['xCoords'], thisRecCurveFeatureInfo['yCoords'])
            except Exception:
                thisRecCurveFeatureInfo.loc[:, 'xIdx'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'yIdx'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'xCoords'] = 0.
                thisRecCurveFeatureInfo.loc[:, 'yCoords'] = 0.
            #
            ampStatsDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'amplitudeStats')
            ampStatsDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'relativeStatsDF')
            relativeStatsDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsNoStimDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'noStimTTest')
            relativeStatsNoStimDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            ampStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'amplitudeStatsPerFreqBand')
            ampStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            relativeStatsPerFBDict[(expName, inputBlockSuffix)] = pd.read_hdf(resultPath, 'relativeStatsDFPerFreqBand')
            relativeStatsPerFBDict[(expName, inputBlockSuffix)].loc[:, 'isMahalDist'] = signalIsMahalDist
            compoundAnnLookupList.append(pd.read_hdf(resultPath, 'compoundAnnLookup'))
            featureInfoList.append(thisRecCurveFeatureInfo)
            ################################################################################################################
            ################################################################################################################
            lOfStatsDFs = [
                ampStatsDict[(expName, inputBlockSuffix)],
                relativeStatsDict[(expName, inputBlockSuffix)],
                # relativeStatsNoStimDict[(expName, inputBlockSuffix)],
                ampStatsPerFBDict[(expName, inputBlockSuffix)],
                relativeStatsPerFBDict[(expName, inputBlockSuffix)]
                ]
            for dfidx, df in enumerate(lOfStatsDFs):
                dfTI = df.index.to_frame().reset_index(drop=True)
                dfTI.loc[:, 'kinematicCondition'] = dfTI['kinematicCondition'].apply(lambda x: x.replace('CCW_', 'CW_'))
                lOfStatsDFs[dfidx].index = pd.MultiIndex.from_frame(dfTI)
            ################################################################################################################
            ################################################################################################################
        except Exception:
            traceback.print_exc()

compoundAnnLookupDF = pd.concat(compoundAnnLookupList).drop_duplicates()
recCurveFeatureInfo = pd.concat(featureInfoList).drop_duplicates()
#recCurveList[0].index.to_frame().reset_index(drop=True)
recCurve = pd.concat(recCurveList)

del recCurveList
ampStatsDF = pd.concat(ampStatsDict, names=['expName', 'selectionName'])
ampStatsDF.drop(labels=['Intercept'], axis='index', level='names', inplace=True)
del ampStatsDict
relativeStatsDF = pd.concat(relativeStatsDict, names=['expName', 'selectionName'])
del relativeStatsDict
relativeStatsNoStimDF = pd.concat(relativeStatsNoStimDict, names=['expName', 'selectionName'])
del relativeStatsNoStimDict
relativeStatsPerFBDF = pd.concat(relativeStatsPerFBDict, names=['expName', 'selectionName'])
del relativeStatsPerFBDict
ampStatsPerFBDF = pd.concat(ampStatsPerFBDict, names=['expName', 'selectionName'])
del ampStatsPerFBDict

########################################################################################################################
########################################################################################################################
'''if (subjectName == 'Murdoc'):
    dfTI = ampStatsDF.index.to_frame().reset_index(drop=True)
    hackMask3 = (
        dfTI['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']) &
        dfTI['electrode'].isin(['-E00+E16', '-E09+E16',])
        ).to_numpy()
    if hackMask3.any():
        ampStatsDF = ampStatsDF.loc[~hackMask3, :]
    dfTI = ampStatsDF.index.to_frame().reset_index(drop=True)
    hackMask4 = (
        dfTI['expName'].isin(['exp201902031100']) &
        dfTI['electrode'].isin(['-E00+E16'])
        ).to_numpy()
    if hackMask4.any():
        ampStatsDF = ampStatsDF.loc[~hackMask4, :]
    dfTI = relativeStatsDF.index.to_frame().reset_index(drop=True)
    hackMask5 = (
        dfTI['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']) &
        dfTI['stimCondition'].isin(['-E00+E16_50.0', '-E09+E16_50.0', '-E00+E16_100.0', '-E09+E16_100.0',])
        ).to_numpy()
    if hackMask5.any():
        relativeStatsDF = relativeStatsDF.loc[~hackMask5, :]
    dfTI = relativeStatsDF.index.to_frame().reset_index(drop=True)
    hackMask6 = (
        dfTI['expName'].isin(['exp201902031100']) &
        dfTI['stimCondition'].isin(['-E00+E16_50.0', '-E00+E16_100.0',])
        ).to_numpy()
    if hackMask6.any():
        relativeStatsDF = relativeStatsDF.loc[~hackMask6, :]
    del dfTI
    '''
#
if (subjectName == 'Rupert') and False:
    dfTI = relativeStatsDF.index.to_frame().reset_index(drop=True)
    hackMask7 = (
            dfTI['expName'].isin(['exp202101251100']) &
            (dfTI['feature'].str.contains('utah_csd_32') | dfTI['feature'].str.contains('utah_csd_74'))
    ).to_numpy()
    if hackMask7.any():
        relativeStatsDF = relativeStatsDF.loc[~hackMask7, :]
    dfTI = ampStatsDF.index.to_frame().reset_index(drop=True)
    hackMask8 = (
            dfTI['expName'].isin(['exp202101251100']) &
            dfTI['feature'].str.contains('utah_csd_32')
    ).to_numpy()
    if hackMask8.any():
        ampStatsDF = ampStatsDF.loc[~hackMask8, :]

########################################################################################################################
########################################################################################################################

correctMultiCompHere = True
confidence_alpha = .05
if correctMultiCompHere:
    pvalsDict = {
        'amp': ampStatsDF.reset_index().loc[:, ['expName', 'pval']],
        'relative': relativeStatsDF.reset_index().loc[:, ['expName', 'pval']],
        'relative_no_stim': relativeStatsNoStimDF.reset_index().loc[:, ['expName', 'pval']],
        }
    pvalsCatDF = pd.concat(pvalsDict, names=['origin', 'originIndex'])
    pvalsCatDF.loc[:, 'reject'] = np.nan
    for gn, pvg in pvalsCatDF.groupby(['origin', 'expName']):
        reject, pval = pg.multicomp(pvg['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
        pvalsCatDF.loc[pvg.index, 'reject'] = reject
        pvalsCatDF.loc[pvg.index, 'pval'] = pval
    '''
    reject, pval = pg.multicomp(pvalsCatDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    pvalsCatDF.loc[:, 'pval'] = pval
    pvalsCatDF.loc[:, 'reject'] = reject
    '''
    dtd = {'pval': float, 'reject': bool}
    for cN in ['pval', 'reject']:
        ampStatsDF.loc[:, cN] = pvalsCatDF.xs('amp', level='origin')[cN].to_numpy(dtype=dtd[cN])
        relativeStatsDF.loc[:, cN] = pvalsCatDF.xs('relative', level='origin')[cN].to_numpy(dtype=dtd[cN])
        relativeStatsNoStimDF.loc[:, cN] = pvalsCatDF.xs('relative_no_stim', level='origin')[cN].to_numpy(dtype=dtd[cN])
    #
    '''
    reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    ampStatsPerFBDF.loc[:, 'pval'] = pval
    ampStatsPerFBDF.loc[:, 'reject'] = reject
    reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    relativeStatsPerFBDF.loc[:, 'pval'] = pval
    relativeStatsPerFBDF.loc[:, 'reject'] = reject
    '''
#q
#
if arguments['freqBandGroup'] == '0':
    freqBandOrderExtended = ['broadband']
elif arguments['freqBandGroup'] == '1':
    freqBandOrderExtended = ['broadband'] + freqBandOrder
elif arguments['freqBandGroup'] == '2':
    freqBandOrderExtended = ['broadband', 'beta', 'gamma']
elif arguments['freqBandGroup'] == '3':
    freqBandOrderExtended = ['gamma', 'higamma']
elif arguments['freqBandGroup'] == '4':
    freqBandOrderExtended = ['broadband', 'beta']
#
akm = ampStatsDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
ampStatsDF = ampStatsDF.loc[akm, :]
rkm = relativeStatsDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
relativeStatsDF = relativeStatsDF.loc[rkm, :]
rnskm = relativeStatsNoStimDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
relativeStatsNoStimDF = relativeStatsNoStimDF.loc[rnskm, :]
rckm = recCurve.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
recCurve = recCurve.loc[rckm, :]
# ampStatsDF, relativeStatsDF, relativeStatsNoStimDF


whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'scaledRAUC'
# whichRAUC = 'dispersion'

figureOutputFolder = os.path.join(
    remoteBasePath, 'figures', 'lfp_recruitment_across_exp')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
#
pdfPath = os.path.join(
    figureOutputFolder, '{}_fbg{}_{}.pdf'.format(subjectName, arguments['freqBandGroup'], pdfNameSuffix))
plotRC = recCurve.reset_index()
plotRC = plotRC.loc[plotRC['electrode'].isin(lOfElectrodesToPlot), :]
ampStatsDF = ampStatsDF.loc[ampStatsDF.index.get_level_values('electrode').isin(lOfElectrodesToPlot), :]
relativeStatsDF = relativeStatsDF.loc[relativeStatsDF.index.get_level_values('electrode').isin(lOfElectrodesToPlot), :]

spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
spinalMapDF.index = pd.Index(spinalMapDF.index, dtype=spinalElecCategoricalDtype)
plotRC.loc[:, 'electrode'] = plotRC['electrode'].astype(spinalElecCategoricalDtype)
presentElectrodes = [eN for eN in lOfElectrodesToPlot if eN in plotRC['electrode'].unique().tolist()]

electrodeErrorBarStyles = {
        'NA': dict(marker='o'),
        '-E00+E16': dict(marker='v'),
        '-E02+E16': dict(marker='s'),
        '-E04+E16': dict(marker='*'),
        '-E05+E16': dict(marker='X'),
        '-E09+E16': dict(marker='d'),
        '-E11+E16': dict(marker='P'),
        '-E12+E16': dict(marker='H'),
        '-E13+E16': dict(marker=r'$\clubsuit$'),
        '-E14+E16': dict(marker='h'),
        }
for eIdx, eN in enumerate(spinalMapDF.index):
    electrodeErrorBarStyles[eN].update(dict(
        elinewidth=sns.plotting_context()['patch.linewidth'],
        capsize=sns.plotting_context()['xtick.major.size']))
#
electrodeErrorBarStyleDF = pd.DataFrame(electrodeErrorBarStyles).loc[:, presentElectrodes]
keepCols = [
    'segment', 'originalIndex', 't',
    'feature', 'freqBandName', 'lag',
    'stimCondition', 'kinematicCondition', 'expName'] + stimulusConditionNames
dropCols = [
    idxName
    for idxName in recCurve.index.names
    if idxName not in keepCols]
plotRC.drop(columns=dropCols, inplace=True)
######

for cN in relativeStatsDF.columns:
    if cN not in relativeStatsNoStimDF.columns:
        relativeStatsNoStimDF.loc[:, cN] = 0
relativeStatsDF.loc[:, 'T_abs'] = relativeStatsDF['T'].abs()
relativeStatsNoStimDF.loc[:, 'T_abs'] = relativeStatsNoStimDF['T'].abs()
#
relativeStatsDF.loc[:, 'cohen_err'] = relativeStatsDF.apply(lambda dfr: (dfr['cohen_ci'][1] - dfr['cohen_ci'][0]) / 2, axis='columns')
relativeStatsNoStimDF.loc[:, 'cohen_err'] = relativeStatsNoStimDF.apply(lambda dfr: (dfr['cohen_ci'][1] - dfr['cohen_ci'][0]) / 2, axis='columns')

ampStatsDF.loc[:, 'coefSign'] = np.sign(ampStatsDF['coef'])
dummyAmpStats = ampStatsDF.copy() * np.nan
#
relativeStatsDF.loc[:, 'hedgesSign'] = np.sign(relativeStatsDF['hedges'])
relativeStatsIndex = relativeStatsDF.index.to_frame().reset_index(drop=True)
relativeStatsIndex.loc[:, 'referenceKinematicCondition'] = relativeStatsIndex['kinematicCondition']
relativeStatsDF.index = pd.MultiIndex.from_frame(relativeStatsIndex)
dummyRelativeStats = relativeStatsDF.copy()
dummyRelativeStats.loc[:, ['T', 'dof', 'pval', 'cohen-d', 'hedges', 'T_abs']] = np.nan

relativeStatsNoStimDF.loc[:, 'hedgesSign'] = np.sign(relativeStatsNoStimDF['hedges'])
relativeStatsNoStimIndex = relativeStatsNoStimDF.index.to_frame().reset_index(drop=True)
#
relativeStatsNoStimIndex.loc[:, 'kinematicCondition'] = relativeStatsNoStimDF['A'].to_numpy()
relativeStatsNoStimIndex.loc[:, 'referenceKinematicCondition'] = relativeStatsNoStimDF['B'].to_numpy()
#
maskForExtVsFlex = ((relativeStatsNoStimDF['A'] != 'NA_NA') & (relativeStatsNoStimDF['B'] != 'NA_NA')).to_numpy()
#
relativeStatsNoStimIndex.loc[maskForExtVsFlex, 'kinematicCondition'] = relativeStatsNoStimDF.loc[maskForExtVsFlex, 'B'].to_numpy()
relativeStatsNoStimIndex.loc[maskForExtVsFlex, 'referenceKinematicCondition'] = relativeStatsNoStimDF.loc[maskForExtVsFlex, 'A'].to_numpy()
relativeStatsNoStimDF.loc[maskForExtVsFlex, 'hedges'] *= -1
relativeStatsNoStimDF.loc[maskForExtVsFlex, 'hedgesSign'] *= -1
relativeStatsNoStimDF.loc[maskForExtVsFlex, 'T'] *= -1
#
relativeStatsNoStimIndex.loc[:, 'electrode'] = 'NA'
relativeStatsNoStimIndex.loc[:, 'kinAndElecCondition'] = relativeStatsNoStimIndex.apply(lambda x: 'NA_{}'.format(x['kinematicCondition']), axis=1)
relativeStatsNoStimIndex.loc[:, 'stimCondition'] = 'NA_0.0'
relativeStatsNoStimIndex.loc[:, 'trialRateInHz'] = 0.

relativeStatsNoStimDF.index = pd.MultiIndex.from_frame(relativeStatsNoStimIndex.loc[:, relativeStatsDF.index.names])
dummyRelativeStatsNoStim = relativeStatsNoStimDF.copy()
dummyRelativeStatsNoStim.loc[:, ['T', 'dof', 'pval', 'cohen-d', 'hedges', 'T_abs']] = np.nan

relevantColumns = [cN for cN in relativeStatsDF.columns if cN in relativeStatsNoStimDF.columns]
refToBaselineMask = relativeStatsNoStimDF.index.get_level_values('referenceKinematicCondition') == 'NA_NA'
#
combinedStatsDF = pd.concat([relativeStatsDF, relativeStatsNoStimDF.loc[:, relevantColumns]])
dummyCombinedStats = combinedStatsDF.copy()
dummyCombinedStats.loc[:, ['T', 'dof', 'pval', 'cohen-d', 'hedges', 'T_abs']] = np.nan
####
countRelativeStatsSignifDF = relativeStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
countAmpStatsSignifDF = ampStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
countRelativeStatsNoStimSignifDF = relativeStatsNoStimDF.groupby(['expName', 'referenceKinematicCondition', 'kinematicCondition']).sum()['reject']
if arguments['verbose']:
    print('Significant amp stats:')
    print(countAmpStatsSignifDF)
    print('Significant relative stats:')
    print(countRelativeStatsSignifDF)

countSummaryHtml = (
    countAmpStatsSignifDF.to_frame(name='significant_stim_effect').to_html() +
    countRelativeStatsSignifDF.to_frame('significant_stim_vs_no_stim').to_html() +
    countRelativeStatsNoStimSignifDF.to_frame('significant_across_movements').to_html()
    )
countSummaryPath = os.path.join(
    figureOutputFolder, '{}_fbg{}_{}.html'.format(subjectName, arguments['freqBandGroup'], pdfNameSuffix))
with open(countSummaryPath, 'w') as _f:
    _f.write(countSummaryHtml)
####
nFeats = plotRC['feature'].unique().shape[0]
nFeatsToPlot = max(min(3, int(np.floor(nFeats/2))), 1)
keepTopIdx = (
    [i for i in range(nFeatsToPlot)] +
    [i for i in range(-1 * nFeatsToPlot, 0)]
    )
keepColsForPlot = []
rankMask = relativeStatsDF.index.get_level_values('stimCondition') != 'NA_0.0'
for freqBandName, relativeStatsThisFB in relativeStatsDF.loc[rankMask, :].groupby('freqBandName'):
    statsRankingDF = relativeStatsThisFB.groupby('feature').mean().sort_values('T', ascending=False)
    nFeatsToPlot = max(min(2, int(np.floor(statsRankingDF.shape[0]/2))), 1)
    keepTopIdx = (
        [i for i in range(nFeatsToPlot)] +
        [i for i in range(-1 * nFeatsToPlot, 0)]
        )
    keepColsForPlot += statsRankingDF.index[keepTopIdx].to_list()
#
if arguments['verbose']:
    print('Plotting select features:')
    print(', '.join(["'{}#0'".format(fN) for fN in keepColsForPlot]))
plotRCPieces = plotRC.loc[plotRC['feature'].isin(keepColsForPlot), :].copy()
######
refGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] == 'NA_0.0', :]
testGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] != 'NA_0.0', :]
#
refGroup = plotRC.loc[plotRC['stimCondition'] == 'NA_0.0', :]
testGroup = plotRC.loc[plotRC['stimCondition'] != 'NA_0.0', :]
#
def genNumSigAnnotator(
        pvDF, xOrder=None, hueVar=None,
        palette=None, fontOpts={},
        textColorMods={},
        width=0.9, nudgeFactor=1.3):
    #
    def numSigAnnotator(g, ro, co, hu, dataSubset):
        if not hasattr(g.axes[ro, co], 'pvalsAnnotated'):
            if not hasattr(g, 'nudgedAxesForPValAnnotation') or (not g._sharey):
                yLim = g.axes[ro, co].get_ylim()
                yExt = yLim[1] - yLim[0]
                g.axes[ro, co].set_ylim([yLim[1] - nudgeFactor * yExt, yLim[1]])
                if not hasattr(g, 'nudgedAxesForPValAnnotation'):
                    g.nudgedAxesForPValAnnotation = True
            trans = transforms.blended_transform_factory(
                g.axes[ro, co].transData, g.axes[ro, co].transAxes)
            hueOrder = palette.index.to_list()
            huePalette = palette.to_dict()
            nHues = max(len(hueOrder), 1)
            offsets = np.linspace(0, width - width / nHues, nHues)
            offsets -= offsets.mean()
            # print(['offsets from {} to {}'.format(offsets[0], offsets[-1])])
            thisMask = pvDF.notna().all(axis='columns')
            if g._col_var is not None:
                thisMask = thisMask & (pvDF[g._col_var] == g.col_names[co])
            if g._row_var is not None:
                thisMask = thisMask & (pvDF[g._row_var] == g.row_names[ro])
            for xIdx, xLabel in enumerate(xOrder):
                xMask = (pvDF[g._x_var] == xLabel)
                for hIdx, hLabel in enumerate(hueOrder):
                    hMask = (pvDF[hueVar] == hLabel)
                    totalMask = (thisMask & xMask & hMask)
                    if totalMask.any():
                        thisEntry = pvDF.loc[totalMask, :]
                        try:
                            assert thisEntry.shape[0] == 1
                        except:
                            traceback.print_exc()
                        thisEntry = thisEntry.iloc[0, :]
                        if (thisEntry.loc[['under', 'ns', 'over']].sum()) == 0:
                            continue
                        message = '{}/{}/{}'.format(
                            int(thisEntry['under']), int(thisEntry['ns']), int(thisEntry['over']))
                        defFontOpts = dict(x=xIdx + offsets[hIdx], y=0.9)
                        defFontOpts.update(fontOpts)
                        if len(textColorMods):
                            thisColor = sns.utils.alter_color(
                                huePalette[hLabel], **textColorMods)
                        else:
                            thisColor = huePalette[hLabel]
                        g.axes[ro, co].text(
                            s=message,
                            transform=trans, color=thisColor,
                            **defFontOpts)
            g.axes[ro, co].pvalsAnnotated = True
        else:
            #  print('ro {} co {} g.axes[ro, co].pvalsAnnotated {}'.format(ro, co, g.axes[ro, co].pvalsAnnotated))
            pass
        return
    return numSigAnnotator

def genNumSigAnnotatorV2(
        pvDF, xOrder=None, hueVar=None,
        palette=None, fontOpts={},
        textColorMods={},
        width=0.9, nudgeFactor=1.3):
    def numSigAnnotator(g, theAx, rowVar, colVar):
        if not hasattr(theAx, 'pvalsAnnotated'):
            axesNeedNudging = (not hasattr(g, 'nudgedAxesForPValAnnotation')) or (not g._sharey)
            if axesNeedNudging:
                yLim = theAx.get_ylim()
                yExt = yLim[1] - yLim[0]
                theAx.set_ylim([yLim[1] - nudgeFactor * yExt, yLim[1]])
                # print('numSigAnnotator: extended ylims to make room')
                if not hasattr(g, 'nudgedAxesForPValAnnotation'):
                    g.nudgedAxesForPValAnnotation = True
            trans = transforms.blended_transform_factory(
                theAx.transData, theAx.transAxes)
            hueOrder = palette.index.to_list()
            huePalette = palette.to_dict()
            nHues = max(len(hueOrder), 1)
            offsets = np.linspace(0, width - width / nHues, nHues)
            offsets -= offsets.mean()
            thisMask = pvDF.notna().all(axis='columns')
            if g._col_var is not None:
                thisMask = thisMask & (pvDF[g._col_var] == colVar)
            if g._row_var is not None:
                thisMask = thisMask & (pvDF[g._row_var] == rowVar)
            for xIdx, xLabel in enumerate(xOrder):
                xMask = (pvDF[g._x_var] == xLabel)
                for hIdx, hLabel in enumerate(hueOrder):
                    hMask = (pvDF[hueVar] == hLabel)
                    totalMask = (thisMask & xMask & hMask)
                    if totalMask.any():
                        thisEntry = pvDF.loc[totalMask, :]
                        assert thisEntry.shape[0] == 1
                        thisEntry = thisEntry.iloc[0, :]
                        if (thisEntry.loc[['under', 'ns', 'over']].sum()) == 0:
                            # print('Warning!\n{}'.format(thisEntry.loc[['under', 'ns', 'over']]))
                            continue
                        message = '{}/{}/{}'.format(
                            int(thisEntry['under']), int(thisEntry['ns']), int(thisEntry['over']))
                        defFontOpts = dict(x=xIdx + offsets[hIdx], y=0.9)
                        defFontOpts.update(fontOpts)
                        if len(textColorMods):
                            thisColor = sns.utils.alter_color(
                                huePalette[hLabel], **textColorMods)
                        else:
                            thisColor = huePalette[hLabel]
                        theAx.text(
                            s=message,
                            transform=trans, color=thisColor,
                            **defFontOpts)
                        # print('numSigAnnotator: annotating axes')
                    else:
                        # print('Warning! No matches for\n{} == {} (columns)\n{} == {}(row)\n{} == {} (hue)'.format(g._col_var, colVar, g._row_var, rowVar, hueVar, hLabel))
                        pass
            theAx.pvalsAnnotated = True
        else:
            # print('numSigAnnotator: axes already annotated')
            pass
        return
    return numSigAnnotator

def genMahalSummaryPlotFun(
        fullDataset=None,
        x=None, order=None, y=None,
        xerr=None, xerr_min=None, xerr_max=None,
        yerr=None, yerr_min=None, yerr_max=None,
        row=None, row_order=None,
        col=None, col_order=None,
        style=None, style_order=None, style_palette=None,
        hue=None, hue_order=None, palette=None,
        error_palette=None,
        dodge=True, width=0.9, jitter=True,
        errorBarOptsDict=None, verbose=False
        ):
    if verbose:
        print('genMahalSummaryPlotFun() width = {}'.format(width))
    semanticNames = [n for n in [row, col, x, hue, style] if n is not None]
    fullDataset.sort_values(semanticNames, inplace=True)
    legendData = {}
    if jitter:
        rng = np.random.default_rng()
    groupingVarList = []
    if row is not None:
        groupingVarList.append(row)
    if col is not None:
        groupingVarList.append(col)
    # for name, group in fullDataset.groupby(groupingVarList):
    if order is not None:
        xOrder = np.asarray(order)
    else:
        xOrder = np.unique(fullDataset[x])
    xMapping = pd.Series(np.arange(xOrder.shape[0]), index=xOrder)
    xMapping.index.name = x
    if hue is not None:
        if hue_order is not None:
            hueOrder = np.asarray(hue_order)
        else:
            hueOrder = np.unique(fullDataset[hue])
        nHues = hueOrder.shape[0]
        if dodge:
            offsets = np.linspace(0, width - width / nHues, nHues)
            offsets -= offsets.mean()
            if verbose:
                print('genMahalSummaryPlotFun() offsets: {}'.format(offsets))
        xHueMappingDict = {}
        legendData[hue] = mpl.patches.Patch(
            alpha=0, linewidth=0)
        for hIdx, hN in enumerate(hueOrder):
            xHueMappingDict[hN] = xMapping.copy()
            if palette is not None:
                legendData[hN] = mpl.patches.Patch(
                    color=palette[hN], linewidth=0)
            if dodge:
                xHueMappingDict[hN] += offsets[hIdx]
        if verbose:
            print('xHueMappingDict:\n{}'.format(pd.concat(xHueMappingDict).sort_values()))
        def xPosFun(dfRow):
            if dfRow[hue] in xHueMappingDict:
                if dfRow[x] in xHueMappingDict[dfRow[hue]]:
                    return xHueMappingDict[dfRow[hue]][dfRow[x]]
            else:
                return np.nan
        if jitter:
            panelGroups = [n for n in [row, col, x, hue] if n is not None]
            xJitters = pd.Series(0, index=fullDataset.index)
            if len(panelGroups):
                grouper = fullDataset.dropna().groupby(panelGroups)
            else:
                grouper = ('all', fullDataset.dropna())
            for name, group in grouper:
                rawJitters = 0.2 * rng.random(group.shape[0]) + 0.8
                rawJitters = np.cumsum(rawJitters) / np.sum(rawJitters)
                scaledJitters = (rawJitters.copy() - 0.5) * width / nHues
                # rng.shuffle(scaledJitters)
                if verbose:
                    print('genMahalSummaryPlotFun()\n    raw jitters min = {} max = {}'.format(
                        scaledJitters.min(), scaledJitters.max()))
                xJitters.loc[group.index] = scaledJitters
            allXPos = fullDataset.apply(xPosFun, axis='columns') + xJitters
            # plt.plot(fullDataset.apply(xPosFun, axis='columns'), xJitters ** 0, 'ro')
            # plt.plot(allXPos, allXPos  ** 0, 'kd'); plt.show()
            # plt.plot(xJitters, xJitters ** 0, 'k.'); plt.show()
        else:
            allXPos = fullDataset.apply(xPosFun, axis='columns')
    else:
        # hue is none
        hueOrder = None
        xHueMappingDict = None
        if jitter:
            rawJitters = 0.5 * rng.random(fullDataset.shape[0]) + 0.5
            rawJitters = np.cumsum(rawJitters) / np.sum(rawJitters)
            rng.shuffle(rawJitters)
            xJitters = pd.Series(
                (rawJitters - 0.5) * width,
                index=fullDataset.index)
            def xPosFun(dfRow):
                if dfRow[x] in xMapping.index:
                    return xMapping[dfRow[x]] + xJitters[dfRow.name]
                else:
                    return np.nan
            allXPos = fullDataset.apply(xPosFun, axis='columns')
        else:
            def xPosFun(dfRow):
                if dfRow[x] in xMapping.index:
                    return xMapping[dfRow[x]]
                else:
                    return np.nan
            allXPos = fullDataset.apply(xPosFun, axis='columns')
    styleOrder = None
    if style is not None:
        if style_order is not None:
            styleOrder = np.asarray(style_order)
        else:
            styleOrder = np.unique(fullDataset[style])
        if isinstance(style_palette, pd.DataFrame):
            legendData[style] = mpl.patches.Patch(alpha=0, linewidth=0)
            if 'marker' in style_palette.index:
                for styleName, thisStyle in style_palette.T.iterrows():
                    mSize = 4. * thisStyle['markersize'] if ('markersize' in thisStyle) else 4. * sns.plotting_context()['lines.markersize']
                    # genMahalSummaryPlotFun(...
                    legendData[styleName] = mpl.lines.Line2D(
                        [0], [0], marker=thisStyle['marker'],
                        markerfacecolor='lightgray', markeredgecolor='k',
                        markeredgewidth=sns.plotting_context()['patch.linewidth'],
                        markersize=mSize, linestyle='none')
    #
    def mahalSummaryPlotFun(
            data=None,
            x=None, order=None, y=None,
            hue=None, hue_order=None, label=None, palette=None,
            color=None, *args, **kwargs):
        ax = plt.gca()
        _ = (
            row, row_order, col, col_order,
            xerr, yerr,
            xerr_min, yerr_min,
            xerr_max, yerr_max,
            allXPos, hueOrder, error_palette, xMapping,
            style, styleOrder, style_palette, errorBarOptsDict)
        inverseXMapping = pd.Series(xMapping.index, index=xMapping.to_numpy())
        nanMask = allXPos.loc[data.index].notna() & data[y].notna()
        goodIndices = data.index[nanMask]
        if style is None:
            grouper = [('all', data.loc[goodIndices, :])]
        else:
            grouper = data.loc[goodIndices, :].groupby(style)
        for styleName, styleGroup in grouper:
            if not styleGroup.empty:
                xx = allXPos.loc[styleGroup.index].to_numpy()
                yy = data.loc[styleGroup.index, y].to_numpy()
                yerrVals = None
                if (yerr_min is not None) or (yerr_max is not None):
                    yerrVals = np.zeros((data.loc[styleGroup.index, :].shape[0], 2))
                    if (yerr_min is not None):
                        yerrVals[:, 0] = data.loc[styleGroup.index, yerr_min].to_numpy()
                    if (yerr_max is not None):
                        yerrVals[:, 1] = data.loc[styleGroup.index, yerr_max].to_numpy()
                    yerrVals = yerrVals.T
                elif yerr is not None:
                    yerrVals = data.loc[styleGroup.index, yerr]
                ebod = dict(ls='none')  # error bar opts dict
                if errorBarOptsDict is not None:
                    ebod.update(errorBarOptsDict)
                if style_palette is not None:
                    if styleName in style_palette:
                        ebod.update(dict(style_palette[styleName]))
                if error_palette is not None:
                    ebod['ecolor'] = error_palette[label]
                ax.errorbar(xx, yy, yerr=yerrVals, color=color, **ebod)
                ax.set_xlim((-0.5, xOrder.shape[0] - 0.5))
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    xTicks = ax.get_xticks()
                    newXTicks = [tL for tL in xTicks if tL in inverseXMapping]
                    newXTickLabels = [inverseXMapping[tL] for tL in xTicks if tL in inverseXMapping]
                    ax.set_xticks(newXTicks)
                    ax.set_xticklabels(newXTickLabels)
        return
    return legendData, mahalSummaryPlotFun

def shareYAcrossRows(
        g=None, yTickerClass=None, yTickerOpts=None):
    axYLimDict = {}
    for ro in range(g.axes.shape[0]):
        for co in range(g.axes.shape[1]):
            axYLimDict[(ro, co)] = g.axes[ro, co].get_ylim()
            if (co != 0) and (g.axes.shape[1] > 1):
                g.axes[ro, 0].get_shared_y_axes().join(g.axes[ro, 0], g.axes[ro, co])
                #
                g.axes[ro, co].set_ylabel(None)
                #
                g.axes[ro, co].set_yticks([])
                g.axes[ro, co].set_yticklabels([])
    axYLimDF = pd.DataFrame(axYLimDict, index=['yMin', 'yMax']).T
    axYLimDF.index.names = ['ro', 'co']
    for ro in range(g.axes.shape[0]):
        newYMin = axYLimDF.xs(ro, level='ro')['yMin'].min()
        newYMax = axYLimDF.xs(ro, level='ro')['yMax'].max()
        # print('setting g.axes[ro, 0].ylim to {:.2f}, {:.2f}'.format(newYMin, newYMax))
        for co in range(g.axes.shape[1]):
            if co == 0:
                g.axes[ro, 0].set_ylim(newYMin, newYMax)
                if yTickerClass is not None:
                    g.axes[ro, 0].yaxis.set_major_locator(yTickerClass(**yTickerOpts))
            else:
                g.axes[ro, co].set_yticks(g.axes[ro, 0].get_yticks())
    return

print('Saving plots to {}'.format(pdfPath))
with PdfPages(pdfPath) as pdf:
    # with contextlib.nullcontext() as pdf:
    gridSpecDefault = dict(wspace=0.05, hspace=0.05)
    rowVar = 'feature'
    colVar = 'stimCondition'
    colOrder = (
        plotRC.loc[
            plotRC['electrode'].isin(lOfElectrodesToPlot),
            ['trialRateInHz', 'electrode', 'stimCondition']]
        .drop_duplicates()
        .sort_values(by=['electrode', 'trialRateInHz'])['stimCondition']
        .to_list())
    # colOrder = np.unique(plotRC[colVar])
    colWrap = min(3, len(colOrder))
    hueName = 'kinematicCondition'
    # hueOrder = sorted(np.unique(plotRC[hueName]))
    hueOrder = ['NA_NA', 'CW_outbound', 'CW_return']
    pal = sns.color_palette("Set2")
    huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
    huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
    if True:
        stripplotKWArgs = dict(
            dodge=True, rasterized=False,
            marker='o',
            )
        goodStripKWArgs = dict(
            edgecolor='k',
            size=sns.plotting_context()['lines.markersize'] / 2,
            linewidth=sns.plotting_context()['patch.linewidth'] / 2)
        badStripKWArgs = dict(
            alpha=0.3, edgecolor='none',
            size=sns.plotting_context()['lines.markersize'] / 4,
            linewidth=0.)
        #
        stripplotKWArgsMahal = stripplotKWArgs.copy()
        stripplotKWArgsMahal['jitter'] = 0.
        stripplotKWArgsMahal['marker'] = 'd'
        goodStripKWArgsMahal = dict(
            edgecolor='k',
            size=1.75 * sns.plotting_context()['lines.markersize'],
            linewidth=sns.plotting_context()['patch.linewidth'])
        badStripKWArgsMahal = dict(
            alpha=0.4, edgecolor='none',
            size=1.75 * sns.plotting_context()['lines.markersize'],
            linewidth=0.)
        boxplotKWArgs = dict(
            dodge=True, whis=np.inf,
            linewidth=sns.plotting_context()['lines.linewidth'],
            )
        boxPlotXTickLabelKWArgs = dict(
            rotation=45, va='top', ha='center')
        nSigAnnFontOpts = dict(
            va='center', ha='center', y=0.125,
            fontsize=sns.plotting_context()["font.size"],
            fontweight='bold', rotation=45)
        #
        boxPlotShareYAcrossRows = True
        shareYAcrossRowsOpts = dict(
            yTickerClass=mpl.ticker.MaxNLocator,
            yTickerOpts=dict(
                nbins=5, steps=[1, 2, 2.5, 5, 10],
                min_n_ticks=3,
                prune='both'))
        #
        colVar = 'electrode'
        rowVar = 'kinematicCondition'
        #
        addMahalGsHere = True
        addChannelGsHere = False
        addNonSigObs = False
        addNonSigObsMahal = True
        #
        enableNumSigAnnotator = True
        figWidthPerBand = 0.125
        figWidthPerBandMahal = 0.15
        figHeightPerRow = 1.
        #
        prettyNameLookup.update({
            'NA_NA': 'Baseline',
            'CW_outbound': 'Extension',
            'CW_return': 'Flexion',
            'NA': 'No stim.'
            })
        ################################################################################################################
        # add dummies to make sure plotting functions allocate all appropriate spaces even if there is no data
        dictWithDummies = {'dummy{}'.format(i): dummyAmpStats.reset_index() for i in range(3)}
        dictWithDummies['original'] = ampStatsDF.reset_index()
        # plotAmpStatsDF = ampStatsDF.reset_index()
        plotAmpStatsDF = pd.concat(dictWithDummies, names=['isDummy', 'originalIndex']).reset_index()
        #
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy0', 'reject'] =        ~ plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy0', 'coefSign'] =        plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'coefSign'].to_numpy()
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy0', 'isMahalDist'] =     plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy1', 'reject'] =        ~ plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy1', 'coefSign'] = (-1) * plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'coefSign'].to_numpy()
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy1', 'isMahalDist'] =     plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy2', 'reject'] =          plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy2', 'coefSign'] = (-1) * plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'coefSign'].to_numpy()
        plotAmpStatsDF.loc[plotAmpStatsDF['isDummy'] == 'dummy2', 'isMahalDist'] =     plotAmpStatsDF.loc[
            plotAmpStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        plotAmpStatsDF.loc[:, 'reject'] = plotAmpStatsDF.loc[:, 'reject'].astype(bool)
        plotAmpStatsDF.loc[:, 'isMahalDist'] = plotAmpStatsDF.loc[:, 'isMahalDist'].astype(bool)
        plotAmpStatsDF.loc[:, 'electrode'] = plotAmpStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotAmpStatsDF.loc[:, 'namesAndMD'] = plotAmpStatsDF['names']
        plotAmpStatsDF.loc[plotAmpStatsDF['isMahalDist'], 'namesAndMD'] += '_md'
        #######
        thisMaskAmp = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) & plotAmpStatsDF['electrode'].isin(lOfElectrodesToPlot) & ~plotAmpStatsDF['isMahalDist']
        if thisMaskAmp.any():
            thisMaskAmpMahal = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) & plotAmpStatsDF['electrode'].isin(lOfElectrodesToPlot) & plotAmpStatsDF['isMahalDist']
            thisFreqBandOrder = [
                fN for fN in freqBandOrderExtended if fN in plotAmpStatsDF.loc[thisMaskAmp, 'freqBandName'].unique().tolist()]
            #
            rowOrder = [rN for rN in lOfKinematicsToPlot if rN in plotAmpStatsDF['kinematicCondition'].to_list()]
            #
            thisPalette = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF.loc[thisMaskAmp, 'namesAndMD'])]
            thisPaletteMahal = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF.loc[thisMaskAmpMahal, 'namesAndMD'])]
            colOrder = [eN for eN in plotAmpStatsDF[colVar].unique().tolist() if ((eN !='NA') and (eN in lOfElectrodesToPlot))]

            if thisPalette.shape[0] > 1:
                stripJitter = 0.9 / (thisPalette.shape[0] + 1)
            else:
                stripJitter = 0.3
            #######################################################################
            # recruitment regression coeffs
            # figHeightPerRow = 1.
            figHeight = figHeightPerRow * (len(rowOrder) + 0.5)
            figWidth = figWidthPerBand * (len(colOrder) * len(freqBandOrderExtended) * thisPalette.shape[0] + 2)
            height = figHeight / len(rowOrder)
            width = figWidth / plotAmpStatsDF.loc[thisMaskAmp, colVar].nunique()
            aspect = width / height
            #
            def countOverUnder(df):
                res = pd.Series({
                    'over': int(((df['coef'] > 0) & df['reject']).sum()),
                    'ns': int((~df['reject']).sum()),
                    'under': int(((df['coef'] < 0) & df['reject']).sum()),
                    })
                return res
            #
            pvalDF = plotAmpStatsDF.loc[thisMaskAmp].dropna().groupby([
                'kinematicCondition', 'electrode', 'freqBandName', 'namesAndMD']).apply(countOverUnder).fillna(0)
            for yVar in ['coefStd']:
                boxPlotMask = (
                    thisMaskAmp &
                    (plotAmpStatsDF['coefSign'] > 0) &
                    plotAmpStatsDF['reject']
                    )
                g = sns.catplot(
                    y=yVar,
                    x='freqBandName', order=thisFreqBandOrder,
                    col=colVar, col_order=colOrder,
                    row=rowVar, row_order=rowOrder,
                    hue='namesAndMD', hue_order=thisPalette.index.to_list(),
                    # palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                    palette=thisPalette.to_dict(), color='w',
                    data=plotAmpStatsDF.loc[boxPlotMask, :],  # & plotAmpStatsDF['reject']
                    height=height, aspect=aspect,
                    sharey=False,
                    sharex=True,  margin_titles=True,
                    facet_kws=dict(gridspec_kws=gridSpecDefault),
                    kind='box', **boxplotKWArgs
                    )
                for name, ax in g.axes_dict.items():
                    rowName, colName = name
                    # non-significant are transparent
                    subSetMask = (
                        thisMaskAmp &
                        (plotAmpStatsDF[rowVar] == rowName) &
                        (plotAmpStatsDF[colVar] == colName) &
                        (~plotAmpStatsDF['reject']))
                    subSetMaskNegative = subSetMask & (plotAmpStatsDF['coefSign'] < 0)
                    subSetMaskNegativeNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    subSetMaskNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    if subSetMaskNotNa.any() and addNonSigObs and addChannelGsHere:
                        sns.stripplot(
                            data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName',
                            order=thisFreqBandOrder,
                            hue='namesAndMD', hue_order=thisPalette.index.to_list(),
                            palette=thisPalette.to_dict(),
                            jitter=stripJitter,
                            **stripplotKWArgs, **badStripKWArgs)
                    # significant are opaque
                    subSetMask = (
                            thisMaskAmp &
                            (plotAmpStatsDF[rowVar] == rowName) &
                            (plotAmpStatsDF[colVar] == colName) &
                            plotAmpStatsDF['reject'])
                    subSetMaskNegative = subSetMask & (plotAmpStatsDF['coefSign'] < 0)
                    subSetMaskNegativeNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    subSetMaskNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    if subSetMaskNegativeNotNa.any():
                        sns.boxplot(
                            data=plotAmpStatsDF.loc[subSetMaskNegative, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue='namesAndMD', hue_order=thisPalette.index.to_list(),
                            # palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                            palette=thisPalette.to_dict(),
                            **boxplotKWArgs
                            )
                    if subSetMaskNotNa.any() and addChannelGsHere:
                        sns.stripplot(
                            data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName',
                            order=thisFreqBandOrder, hue='namesAndMD',
                            hue_order=thisPalette.index.to_list(),
                            palette=thisPalette.to_dict(),
                            jitter=stripJitter,
                            **stripplotKWArgs, **goodStripKWArgs
                            )
                    # non-significant are transparent (mahal)
                    subSetMaskMahal = (
                            thisMaskAmpMahal &
                            (plotAmpStatsDF[rowVar] == rowName) &
                            (plotAmpStatsDF[colVar] == colName) &
                            (~plotAmpStatsDF['reject']))
                    subSetMaskMahalNegative = subSetMask & (plotAmpStatsDF['coefSign'] < 0)
                    subSetMaskMahalNegativeNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    subSetMaskMahalNotNa = subSetMaskMahal & plotAmpStatsDF[yVar].notna()
                    if subSetMaskMahalNotNa.any() and addMahalGsHere and addNonSigObsMahal:
                        sns.stripplot(
                            data=plotAmpStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName',
                            order=thisFreqBandOrder,
                            hue='namesAndMD', hue_order=thisPaletteMahal.index.to_list(),
                            palette=thisPaletteMahal.to_dict(),
                            **stripplotKWArgsMahal, **badStripKWArgsMahal)
                    # significant are opaque (mahal)
                    subSetMaskMahal = (
                        thisMaskAmpMahal &
                        (plotAmpStatsDF[rowVar] == rowName) &
                        (plotAmpStatsDF[colVar] == colName) &
                        plotAmpStatsDF['reject'])
                    subSetMaskMahalNegative = subSetMask & (plotAmpStatsDF['coefSign'] < 0)
                    subSetMaskMahalNegativeNotNa = subSetMask & plotAmpStatsDF[yVar].notna()
                    subSetMaskMahalNotNa = subSetMaskMahal & plotAmpStatsDF[yVar].notna()
                    if subSetMaskMahalNotNa.any() and addMahalGsHere:
                        sns.stripplot(
                            data=plotAmpStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName',
                            order=thisFreqBandOrder, hue='namesAndMD',
                            hue_order=thisPaletteMahal.index.to_list(), palette=thisPaletteMahal.to_dict(),
                            **stripplotKWArgsMahal, **goodStripKWArgsMahal
                            )
                    xTickLabels = ax.get_xticklabels()
                    if len(xTickLabels):
                        newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                        ax.set_xticklabels( newXTickLabels, **boxPlotXTickLabelKWArgs)
                    ax.axhline(0, c="0.5", ls='--', zorder=0.9)
                    for xJ in range(1, len(thisFreqBandOrder), 2):
                        ax.axvspan(
                            -0.5 + xJ, 0.5 + xJ,
                            color="0.1", alpha=0.1, zorder=1.)
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                    ax.set_xlim([-0.5, len(thisFreqBandOrder) - 0.5])
                g.set_titles(col_template='{col_name}', row_template='{row_name}')
                plotProcFuns = [asp.genTitleChanger(prettyNameLookup)]
                if enableNumSigAnnotator:
                    plotProcFuns += [
                    genNumSigAnnotator(
                        pvalDF.reset_index(),
                        xOrder=thisFreqBandOrder,
                        hueVar='namesAndMD', palette=thisPalette,
                        # width=boxplotKWArgs['width'],
                        fontOpts=nSigAnnFontOpts,
                        textColorMods=dict(l=-0.5)
                        )]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                # g.suptitle('Coefficient distribution for AUC regression')
                asp.reformatFacetGridLegend(
                    g, titleOverrides={
                        'namesAndMD': 'Regressors'
                    },
                    contentOverrides={
                        'namesAndMD': 'Regressors',
                        'trialAmplitude': 'Stimulation\namplitude',
                        'trialAmplitude:trialRateInHz': 'Stimulation-rate\ninteraction',
                        'trialAmplitude_md': 'Stimulation\namplitude\n(Mahal. dist.)',
                        'trialAmplitude:trialRateInHz_md': 'Stimulation-rate\ninteraction\n(Mahal. dist.)',
                    },
                styleOpts=styleOpts)
                g.set_axis_labels(
                    prettyNameLookup['freqBandName'],
                    prettyNameLookup[yVar])
                g.resize_legend(adjust_subtitles=True)
                #
                if boxPlotShareYAcrossRows:
                    shareYAcrossRows(g, **shareYAcrossRowsOpts)
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                ########################
                if not consoleDebug:
                    pdf.savefig(bbox_inches='tight', pad_inches=0,)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            ######
        ################################################################################################################
        # add dummies to make sure plotting functions allocate all appropriate spaces even if there is no data
        dictWithDummies = {'dummy{}'.format(i): dummyCombinedStats.reset_index() for i in range(3)}
        dictWithDummies['original'] = combinedStatsDF.reset_index()
        # plotRelativeStatsDF = ampStatsDF.reset_index()
        plotRelativeStatsDF = pd.concat(dictWithDummies, names=['isDummy', 'originalIndex']).reset_index()
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy0', 'reject'] = ~ plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy0', 'hedgesSign'] = plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'hedgesSign'].to_numpy()
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy0', 'isMahalDist'] = plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy1', 'reject'] = ~ plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy1', 'hedgesSign'] = (-1) * plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'hedgesSign'].to_numpy()
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy1', 'isMahalDist'] = plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy2', 'reject'] = plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'reject'].to_numpy(dtype=bool)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy2', 'hedgesSign'] = (-1) * plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'hedgesSign'].to_numpy()
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isDummy'] == 'dummy2', 'isMahalDist'] = plotRelativeStatsDF.loc[
            plotRelativeStatsDF['isDummy'] == 'original', 'isMahalDist'].to_numpy(dtype=bool)
        #
        crossMovementComparisons = (
            (plotRelativeStatsDF['electrode'] == 'NA') &
            (plotRelativeStatsDF['referenceKinematicCondition'] != 'NA_NA')).to_numpy()
        #
        plotRelativeStatsDF.loc[:, 'reject'] = plotRelativeStatsDF.loc[:, 'reject'].astype(bool)
        plotRelativeStatsDF.loc[:, 'isMahalDist'] = plotRelativeStatsDF.loc[:, 'isMahalDist'].astype(bool)
        #
        plotRelativeStatsDF.loc[:, 'trialRateInHzStr'] = plotRelativeStatsDF['trialRateInHz'].apply(lambda x: '{}'.format(x))
        plotRelativeStatsDF.loc[:, 'electrode'] = plotRelativeStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isMahalDist'], 'trialRateInHzStr'] += '_md'

        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(x['feature'].replace('#0', '')), axis='columns')
        ###
        lOfStimElectrodesToPlot = [eN for eN in lOfElectrodesToPlot if eN != 'NA']
        thisMaskRel = (
            plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) &
            (plotRelativeStatsDF['electrode'].isin(lOfStimElectrodesToPlot)) &
            ~plotRelativeStatsDF['isMahalDist'])
        #
        thisMaskRelMahal = (
            plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) &
            plotRelativeStatsDF['electrode'].isin(lOfStimElectrodesToPlot) &
            plotRelativeStatsDF['isMahalDist'])
        #
        thisFreqBandOrder = [
            fN
            for fN in freqBandOrderExtended
            if fN in plotRelativeStatsDF.loc[thisMaskRel, 'freqBandName'].unique().tolist()]
        ####
        thisPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRel, 'trialRateInHzStr'])]
        ######
        thisPaletteMahal = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelMahal, 'trialRateInHzStr'])]
        colOrder = [
            eN for eN in plotRelativeStatsDF[colVar].sort_values().unique().tolist() if (eN in lOfStimElectrodesToPlot)]

        stripHue = 'trialRateInHzStr'
        stripHueOrder = thisPalette.index.to_list()
        stripPalette = thisPalette.to_dict()
        stripPaletteMahal = thisPaletteMahal.to_dict()
        stripHueOrderMahal = thisPaletteMahal.index.to_list()
        stripJitter = 0.9 / (len(stripHueOrder) + 1)

        def countOverUnder(df):
            res = pd.Series({
                'over': int(((df['T'] > 0) & df['reject']).sum()),
                'ns': int((~df['reject']).sum()),
                'under': int(((df['T'] < 0) & df['reject']).sum()),
            })
            return res
        #
        pvalDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby([
            'kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).apply(countOverUnder).fillna(0)
        ### effect size cat plot stripplot
        figHeight = figHeightPerRow * (len(rowOrder) + 0.5)
        figWidth = figWidthPerBand * (len(colOrder) * len(freqBandOrderExtended) * len(stripHueOrder) + 2)
        height = figHeight / plotRelativeStatsDF.loc[thisMaskRel, rowVar].nunique()
        width = figWidth / plotRelativeStatsDF.loc[thisMaskRel, colVar].nunique()
        aspect = width / height
        figTitleStr = 'Effect size distribution for stim vs no-stim comparisons'
        #
        def plotThisThing():
            boxPlotMask = (
                thisMaskRel &
                (plotRelativeStatsDF['hedgesSign'] > 0) &
                plotRelativeStatsDF['reject']
                )
            for yVar in ['hedges']:
                g = sns.catplot(
                    y=yVar,
                    x='freqBandName',
                    order=thisFreqBandOrder,
                    col=colVar, col_order=colOrder,
                    row=rowVar, row_order=rowOrder,
                    height=height, aspect=aspect,
                    sharey=False, sharex=True, margin_titles=True,
                    hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                    # palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                    palette=thisPalette.to_dict(),
                    data=plotRelativeStatsDF.loc[boxPlotMask, :],
                    kind='box', facet_kws=dict(gridspec_kws=gridSpecDefault),
                    **boxplotKWArgs
                    )
                for name, ax in g.axes_dict.items():
                    rowName, colName = name
                    # plot non-significant observations with transparency
                    subSetMask = (
                            (plotRelativeStatsDF[rowVar] == rowName) &
                            (plotRelativeStatsDF[colVar] == colName) &
                            (~plotRelativeStatsDF['reject']) &
                            thisMaskRel)
                    subSetMaskNegative = subSetMask & (plotRelativeStatsDF['hedgesSign'] < 0)
                    subSetMaskNegativeNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    subSetMaskNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    if subSetMaskNotNa.any() and addNonSigObs and addChannelGsHere:
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrder, palette=stripPalette,
                            jitter=stripJitter,
                            **stripplotKWArgs, **badStripKWArgs)
                    # plot significant observations fully opaque
                    subSetMask = (
                            (plotRelativeStatsDF[rowVar] == rowName) &
                            (plotRelativeStatsDF[colVar] == colName) &
                            plotRelativeStatsDF['reject'] & thisMaskRel)
                    subSetMaskNegative = subSetMask & (plotRelativeStatsDF['hedgesSign'] < 0)
                    subSetMaskNegativeNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    subSetMaskNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    if subSetMaskNegativeNotNa.any():
                        sns.boxplot(
                            data=plotRelativeStatsDF.loc[subSetMaskNegative, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                            # palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                            palette=thisPalette.to_dict(),
                            **boxplotKWArgs
                            )
                    if subSetMaskNotNa.any() and addChannelGsHere:
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrder, palette=stripPalette,
                            jitter=stripJitter,
                            **stripplotKWArgs, **goodStripKWArgs
                            )
                    #####
                    # plot non-significant observations with transparency (mahal)
                    subSetMaskMahal = (
                        (plotRelativeStatsDF[rowVar] == rowName) &
                        (plotRelativeStatsDF[colVar] == colName) &
                        (~plotRelativeStatsDF['reject']) &
                        thisMaskRelMahal)
                    subSetMaskMahalNegative = subSetMask & (plotRelativeStatsDF['hedgesSign'] < 0)
                    subSetMaskMahalNegativeNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    subSetMaskMahalNotNa = subSetMaskMahal & plotRelativeStatsDF[yVar].notna()
                    if subSetMaskMahalNotNa.any() and addMahalGsHere and addNonSigObsMahal:
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrderMahal, palette=stripPaletteMahal,
                            **stripplotKWArgsMahal, **badStripKWArgsMahal)
                    # plot significant observations fully opaque (mahal)
                    subSetMaskMahal = (
                            (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                            plotRelativeStatsDF['reject'] & thisMaskRelMahal)
                    subSetMaskMahalNegative = subSetMask & (plotRelativeStatsDF['hedgesSign'] < 0)
                    subSetMaskMahalNegativeNotNa = subSetMask & plotRelativeStatsDF[yVar].notna()
                    subSetMaskMahalNotNa = subSetMaskMahal & plotRelativeStatsDF[yVar].notna()
                    if subSetMaskMahalNotNa.any() and addMahalGsHere:
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrderMahal, palette=stripPaletteMahal,
                            **stripplotKWArgsMahal, **goodStripKWArgsMahal
                            )
                    nSigAnnotator = genNumSigAnnotatorV2(
                        pvalDF.reset_index(),
                        xOrder=thisFreqBandOrder,
                        hueVar='trialRateInHzStr', palette=thisPalette,
                        # width=boxplotKWArgs['width'],
                        fontOpts=nSigAnnFontOpts,
                        textColorMods=dict(l=-0.5))
                    if enableNumSigAnnotator:
                        nSigAnnotator(g, ax, rowName, colName)
                    ax.axhline(0, c="0.5", ls='--', zorder=0.9)
                    xTickLabels = ax.get_xticklabels()
                    if len(xTickLabels):
                        newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                        ax.set_xticklabels(newXTickLabels, **boxPlotXTickLabelKWArgs)
                    for xJ in range(1, len(thisFreqBandOrder), 2):
                        ax.axvspan(-0.5 + xJ, 0.5 + xJ, color="0.1", alpha=0.1, zorder=1.)
                    if (ax.get_legend() is not None):
                        ax.get_legend().remove()
                    ax.set_xlim([-0.5, len(thisFreqBandOrder) - 0.5])
                g.set_titles(col_template='{col_name}', row_template='{row_name}')
                plotProcFuns = [
                    asp.genTitleChanger(prettyNameLookup),
                    ]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                # g.suptitle(figTitleStr)
                asp.reformatFacetGridLegend(
                    g, titleOverrides=prettyNameLookup,
                    contentOverrides=prettyNameLookup,
                styleOpts=styleOpts)
                g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
                g.resize_legend(adjust_subtitles=True)
                if boxPlotShareYAcrossRows:
                    shareYAcrossRows(g, **shareYAcrossRowsOpts)
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                if not consoleDebug:
                    pdf.savefig(bbox_inches='tight', pad_inches=0,)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            return
        #
        plotThisThing()
        #
        thisMaskRel = (
                plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) &
                (plotRelativeStatsDF['electrode'].isin(['NA'])) &
                ~plotRelativeStatsDF['isMahalDist'])
        thisMaskRelMahal = (
                plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) &
                plotRelativeStatsDF['electrode'].isin(['NA']) &
                plotRelativeStatsDF['isMahalDist'])
        if (arguments['freqBandGroup'] == '0') and False:
            thisMaskRel = thisMaskRel & (plotRelativeStatsDF['expName'].isin(['exp202101211100']))
            thisMaskRelMahal = thisMaskRelMahal & (plotRelativeStatsDF['expName'].isin(['exp202101211100']))
        thisFreqBandOrder = [
            fN
            for fN in freqBandOrderExtended
            if fN in plotRelativeStatsDF.loc[thisMaskRel, 'freqBandName'].unique().tolist()]
        ####
        thisPalette = allRelPalette.loc[
            allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRel, 'trialRateInHzStr'])]
        ######
        thisPaletteMahal = allRelPalette.loc[
            allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelMahal, 'trialRateInHzStr'])]
        #
        colVar = 'kinematicCondition'
        colOrder = [
            cN
            for cN in lOfKinematicsToPlot
            if cN in plotRelativeStatsDF.loc[thisMaskRel, colVar].to_list()]
        rowVar = 'referenceKinematicCondition'
        rowOrder = [
            cN
            for cN in ['NA_NA', 'CW_outbound', 'CW_return']
            if cN in plotRelativeStatsDF.loc[thisMaskRel, rowVar].to_list()]

        stripHue = 'trialRateInHzStr'
        stripHueOrder = thisPalette.index.to_list()
        stripPalette = thisPalette.to_dict()
        stripPaletteMahal = thisPaletteMahal.to_dict()
        stripHueOrderMahal = thisPaletteMahal.index.to_list()

        if thisPalette.shape[0] > 1:
            stripJitter = .9 / (thisPalette.shape[0] + 1)
        else:
            stripJitter = .3
        #
        #
        pvalDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(
            [rowVar, colVar, 'freqBandName', 'trialRateInHzStr']).apply(countOverUnder).fillna(0)
        figHeight = figHeightPerRow * (len(rowOrder) + 0.5)
        figWidth = figWidthPerBand * (len(colOrder) * len(freqBandOrderExtended) + 2)
        #
        height = figHeight / len(rowOrder)
        width = figWidth / plotRelativeStatsDF.loc[thisMaskRel, colVar].nunique()
        aspect = width / height
        figTitleStr = 'Effect size distribution across movement comparisons'
        #
        plotThisThing()
        #####
        ######## mahal dist plots
        plotMahalDistSummary = True
        if plotMahalDistSummary:
            thisMaskRel = (
                ~crossMovementComparisons &
                plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) &
                plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot) &
                (plotRelativeStatsDF['isDummy'] == 'original') &
                plotRelativeStatsDF['isMahalDist'])
        if (arguments['freqBandGroup'] == '0') and False:
            thisMaskRel = thisMaskRel & (plotRelativeStatsDF['expName'].isin(['exp202101211100']))
            thisMaskRelMahal = thisMaskRelMahal & (plotRelativeStatsDF['expName'].isin(['exp202101211100']))
            #
        rowVar = 'kinematicCondition'
        rowOrder = [
            cN
            for cN in lOfKinematicsToPlot
            if cN in plotRelativeStatsDF.loc[thisMaskRel, rowVar].to_list()]
        #
        ######## mahal dist plots
        figHeight = figHeightPerRow * (len(rowOrder) + 0.5)
        figWidth = figWidthPerBandMahal * (len(freqBandOrderExtended) * len(lOfElectrodesToPlot) + 2)
        height = figHeight / len(rowOrder)
        width = figWidth
        aspect = width / height
        #
        if thisMaskRel.any():
            thisPalette = allRelPalette.loc[
                allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRel, 'trialRateInHzStr'])]
            thisErrPalette = thisPalette.apply(sns.utils.alter_color, l=-0.5)
            yVar = 'hedges'
            g = sns.FacetGrid(
                ##y=yVar,
                ### x=colVar, order=colOrder,
                ##x='freqBandName', order=thisFreqBandOrder,
                row=rowVar, row_order=rowOrder,
                height=height, aspect=aspect,
                sharey=False, sharex=True, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                palette=thisPalette.to_dict(),
                gridspec_kws=gridSpecDefault,
                data=plotRelativeStatsDF.loc[thisMaskRel, :])
            #
            legendData, mDSummaryPlotFun = genMahalSummaryPlotFun(
                fullDataset=plotRelativeStatsDF.loc[thisMaskRel, :].copy(),
                x='freqBandName', order=thisFreqBandOrder, y=yVar,
                row=rowVar, row_order=rowOrder,
                hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                style='electrode', style_palette=electrodeErrorBarStyleDF,
                yerr='cohen_err', error_palette=thisErrPalette.to_dict(),
                palette=thisPalette.to_dict(),
                errorBarOptsDict=dict(
                    markeredgecolor='k',
                    markeredgewidth=sns.plotting_context()['patch.linewidth'],
                    markersize=2.5 * sns.plotting_context()['lines.markersize'],
                    capsize=sns.plotting_context()['xtick.major.size'])
                    )
            mapDFProcFuns = [
                (
                    mDSummaryPlotFun, [], dict(
                        x='freqBandName', order=thisFreqBandOrder, y=yVar,
                        hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                        palette=thisPalette.to_dict())),
                ]
            for mpdf in mapDFProcFuns:
                mpdf_fun, mpdf_args, mpdf_kwargs = mpdf
                g.map_dataframe(mpdf_fun, *mpdf_args, **mpdf_kwargs)
            for name, ax in g.axes_dict.items():
                rowName = name
                ax.axhline(0, c="0.5", ls='--', zorder=0.9)
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                    ax.set_xticklabels(newXTickLabels, **boxPlotXTickLabelKWArgs)
                for xJ in range(1, len(thisFreqBandOrder), 2):
                    # for xJ in range(0, plotRelativeStatsDF[colVar].unique().shape[0], 2):
                    ax.axvspan(-0.5 + xJ, 0.5 + xJ, color="0.1", alpha=0.1, zorder=1.)
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
                ax.set_xlim([-0.5, len(thisFreqBandOrder) - 0.5])
            g.set_titles(col_template='{col_name}', row_template='{row_name}')
            plotProcFuns = [asp.genTitleChanger(prettyNameLookup),]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            # g.suptitle(
            #     'Effect size distribution for stim vs no-stim comparisons (Mahal dist)')
            g.add_legend(legend_data=legendData, label_order=list(legendData.keys()))
            asp.reformatFacetGridLegend(
                g, titleOverrides=prettyNameLookup,
                contentOverrides=prettyNameLookup,
                styleOpts=styleOpts)
            g.set_axis_labels(
                prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            if boxPlotShareYAcrossRows:
                shareYAcrossRows(g, **shareYAcrossRowsOpts)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            if not consoleDebug:
                pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
        #####################################################################
        if True:
            # dummy plot for legends
            g = sns.catplot(
                y=yVar,
                x=colVar, order=colOrder,
                height=height, aspect=aspect,
                sharey=False, sharex=True, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                # palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                palette=thisPalette.to_dict(),
                data=plotRelativeStatsDF, kind='box',
                facet_kws=dict(gridspec_kws=gridSpecDefault),
                **boxplotKWArgs)
            # g.suptitle('Dummy plot to get full legend from')
            legendData = {}
            legendData['coefStd'] = mpl.patches.Patch(alpha=0, linewidth=0)
            for factorName, factorColor in allAmpPalette.items():
                if '_md' in factorName:
                    legendData[factorName] = mpl.lines.Line2D(
                        [0], [0], marker='d',
                        markerfacecolor=factorColor, markeredgecolor='k',
                        markeredgewidth=sns.plotting_context()['patch.linewidth'],
                        markersize=3. * sns.plotting_context()['lines.markersize'],
                        linestyle='none')
                else:
                    legendData[factorName] = mpl.lines.Line2D(
                        [0], [0], marker='d',
                        markerfacecolor=factorColor, markeredgecolor='k',
                        markeredgewidth=sns.plotting_context()['patch.linewidth'],
                        markersize=2. * sns.plotting_context()['lines.markersize'],
                        linestyle='none')
            legendData['trialRateInHzStr'] = mpl.patches.Patch(alpha=0, linewidth=0)
            for rateName, rateColor in allRelPalette.items():
                if '_md' in rateName:
                    legendData[rateName] = mpl.lines.Line2D(
                        [0], [0], marker='d',
                        markerfacecolor=rateColor, markeredgecolor='k',
                        markeredgewidth=sns.plotting_context()['patch.linewidth'],
                        markersize=3. * sns.plotting_context()['lines.markersize'],
                        linestyle='none')
                else:
                    legendData[rateName] = mpl.lines.Line2D(
                        [0], [0], marker='o',
                        markerfacecolor=rateColor, markeredgecolor='k',
                        markeredgewidth=sns.plotting_context()['patch.linewidth'],
                        markersize=2. * sns.plotting_context()['lines.markersize'],
                        linestyle='none')
            legendData['p-value significance'] = mpl.patches.Patch(alpha=0, linewidth=0)
            legendData['p < {:.2f}'.format(confidence_alpha)] = mpl.lines.Line2D(
                [0], [0], marker='d',
                markerfacecolor='0.1', markeredgecolor='k',
                markeredgewidth=sns.plotting_context()['patch.linewidth'],
                markersize=3. * sns.plotting_context()['lines.markersize'],
                linestyle='none')
            legendData['p > {:.2f} (n.s.)'.format(confidence_alpha)] = mpl.lines.Line2D(
                [0], [0], marker='d',
                markerfacecolor='0.9', markeredgecolor='k',
                markeredgewidth=sns.plotting_context()['patch.linewidth'],
                markersize=3. * sns.plotting_context()['lines.markersize'],
                linestyle='none')
            g.add_legend(
                legend_data=legendData, label_order=list(legendData.keys()), title='dummy legends')
            asp.reformatFacetGridLegend(
                g, titleOverrides=prettyNameLookup, contentOverrides=prettyNameLookup,
            styleOpts=styleOpts)
            g.set_axis_labels(prettyNameLookup[colVar], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            if not consoleDebug:
                pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
#
print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
