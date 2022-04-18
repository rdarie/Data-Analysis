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
snsContext = 'talk'
snsRCParams = {
        'axes.facecolor': 'w',
        "xtick.direction": 'in',
        "ytick.direction": 'in',
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.spines.right": True,
        "axes.spines.top": True,
        "xtick.bottom": True,
        "xtick.top": True,
        "ytick.left": True,
        "ytick.right": True,
        #
    }
snsRCParams.update(customSeabornContexts[snsContext])
mplRCParams = {
    'figure.dpi': useDPI, 'savefig.dpi': useDPI,
    #
    'axes.titlepad': 1.5,
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
    'panel_heading.pad': 1.
    }

for rcK, rcV in mplRCParams.items():
    mpl.rcParams[rcK] = rcV

sns.set(
    context=snsContext, style='whitegrid',
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
elif arguments['plotSuffix'] == 'E04':
    pdfNameSuffix = 'RAUC_E04'
    lOfKinematicsToPlot = ['NA_NA', 'CCW_outbound', 'CW_outbound']
elif arguments['plotSuffix'] == 'rest_E04':
    pdfNameSuffix = 'RAUC_rest_E04'
    lOfKinematicsToPlot = ['NA_NA']
elif arguments['plotSuffix'] == 'move_E09':
    pdfNameSuffix = 'RAUC_move_E09'
    lOfKinematicsToPlot = ['CW_outbound', 'CW_return', 'CCW_outbound', 'CCW_return']
elif arguments['plotSuffix'] == 'outbound':
    pdfNameSuffix = 'RAUC_outbound'
    lOfKinematicsToPlot = ['NA_NA', 'CW_outbound', 'CW_return', 'CCW_outbound', 'CCW_return']
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
        elif arguments['plotSuffix'] == 'rest_E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'E04':
            lOfElectrodesToPlot = [eN for eN in ['NA', '-E04+E16'] if eN in spinalMapDF.index]
        elif arguments['plotSuffix'] == 'outbound':
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
            thisRecCurve.index = pd.MultiIndex.from_frame(recCurveTrialInfo)
            hackMask1 = recCurveTrialInfo['expName'].isin(['exp201901251000', 'exp201901261000', 'exp201901271000']).any()
            if hackMask1:
                dropMask = recCurveTrialInfo['electrode'].isin(['-E09+E16', '-E00+E16']).to_numpy()
                thisRecCurve = thisRecCurve.loc[~dropMask, :]
            hackMask2 = recCurveTrialInfo['expName'].isin(['exp201902031100']).any()
            if hackMask2:
                dropMask = recCurveTrialInfo['electrode'].isin(['-E00+E16']).to_numpy()
                thisRecCurve = thisRecCurve.loc[~dropMask, :]
            ################################################################################################################
            ################################################################################################################
            thisRecCurve.loc[:, 'isMahalDist'] = signalIsMahalDist
            thisRecCurveFeatureInfo.loc[:, 'isMahalDist'] = signalIsMahalDist
            recCurveList.append(thisRecCurve)
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
#
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
#
if subjectName == 'Rupert':
    relativeStatsDF.xs(['exp202101251100', 'broadband'], level=['expName', 'freqBandName']).sort_values('hedges')
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
        'amp': ampStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative': relativeStatsDF.loc[:, ['pval']].reset_index(drop=True),
        'relative_no_stim': relativeStatsNoStimDF.loc[:, ['pval']].reset_index(drop=True),
        }
    pvalsCatDF = pd.concat(pvalsDict, names=['origin', 'originIndex'])
    reject, pval = pg.multicomp(pvalsCatDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    pvalsCatDF.loc[:, 'pval'] = pval
    pvalsCatDF.loc[:, 'reject'] = reject
    for cN in ['pval', 'reject']:
        ampStatsDF.loc[:, cN] = pvalsCatDF.xs('amp', level='origin')[cN].to_numpy()
        relativeStatsDF.loc[:, cN] = pvalsCatDF.xs('relative', level='origin')[cN].to_numpy()
        relativeStatsNoStimDF.loc[:, cN] = pvalsCatDF.xs('relative_no_stim', level='origin')[cN].to_numpy()
    reject, pval = pg.multicomp(ampStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    ampStatsPerFBDF.loc[:, 'pval'] = pval
    ampStatsPerFBDF.loc[:, 'reject'] = reject
    reject, pval = pg.multicomp(relativeStatsPerFBDF['pval'].to_numpy(), alpha=confidence_alpha, method='fdr_bh')
    relativeStatsPerFBDF.loc[:, 'pval'] = pval
    relativeStatsPerFBDF.loc[:, 'reject'] = reject
#q
#
if arguments['freqBandGroup'] == '0':
    freqBandOrderExtended = ['broadband']
elif arguments['freqBandGroup'] == '1':
    freqBandOrderExtended = ['broadband'] + freqBandOrder
elif arguments['freqBandGroup'] == '2':
    freqBandOrderExtended = ['broadband', 'beta', 'gamma']
elif arguments['freqBandGroup'] == '3':
    freqBandOrderExtended = ['gamma']
elif arguments['freqBandGroup'] == '4':
    freqBandOrderExtended = ['beta']
akm = ampStatsDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
ampStatsDF = ampStatsDF.loc[akm, :]
rkm = relativeStatsDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
relativeStatsDF = relativeStatsDF.loc[rkm, :]
rnskm = relativeStatsNoStimDF.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
relativeStatsNoStimDF = relativeStatsNoStimDF.loc[rnskm, :]
rckm = recCurve.index.get_level_values('freqBandName').isin(freqBandOrderExtended)
recCurve = recCurve.loc[rckm, :]
# ampStatsDF, relativeStatsDF, relativeStatsNoStimDF

countRelativeStatsSignifDF = relativeStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
countAmpStatsSignifDF = ampStatsDF.groupby(['expName', 'electrode', 'kinematicCondition']).sum()['reject']
if arguments['verbose']:
    print('Significant amp stats:')
    print(countAmpStatsSignifDF)
    print('Significant relative stats:')
    print(countRelativeStatsSignifDF)

whichRAUC = 'rawRAUC'
# whichRAUC = 'normalizedRAUC'
# whichRAUC = 'scaledRAUC'
# whichRAUC = 'dispersion'
figureOutputFolder = os.path.join(
    remoteBasePath, 'figures', 'lfp_recruitment_across_exp')
if not os.path.exists(figureOutputFolder):
    os.makedirs(figureOutputFolder, exist_ok=True)
#
countSummaryHtml = (
        countAmpStatsSignifDF.to_frame(name='significant_stim_effect').to_html() +
        countRelativeStatsSignifDF.to_frame('significant_stim_vs_no_stim').to_html())
countSummaryPath = os.path.join(
    figureOutputFolder, '{}_fbg{}_{}.html'.format(subjectName, arguments['freqBandGroup'], pdfNameSuffix))
with open(countSummaryPath, 'w') as _f:
    _f.write(countSummaryHtml)
#
pdfPath = os.path.join(
    figureOutputFolder, '{}_fbg{}_{}.pdf'.format(subjectName, arguments['freqBandGroup'], pdfNameSuffix))
plotRC = recCurve.reset_index()
plotRC = plotRC.loc[plotRC['electrode'].isin(lOfElectrodesToPlot), :]
ampStatsDF = ampStatsDF.loc[ampStatsDF.index.get_level_values('electrode').isin(lOfElectrodesToPlot), :]
relativeStatsDF = relativeStatsDF.loc[relativeStatsDF.index.get_level_values('electrode').isin(lOfElectrodesToPlot), :]
# lOfElectrodesToPlot
spinalElecCategoricalDtype = pd.CategoricalDtype(spinalMapDF.index.to_list(), ordered=True)
spinalMapDF.index = pd.Index(spinalMapDF.index, dtype=spinalElecCategoricalDtype)
plotRC.loc[:, 'electrode'] = plotRC['electrode'].astype(spinalElecCategoricalDtype)
presentElectrodes = [eN for eN in lOfElectrodesToPlot if eN in plotRC['electrode'].unique().tolist()]
# plotRC.groupby('electrode')['trialAmplitude'].max().dropna()

# allFilledMarkers = mpl.lines.Line2D.filled_markers
allFilledMarkers = ('o', '*', 'D', 'X', 'P', 's', 'v', 'H')
electrodeErrorBarStyles = {}
for eIdx, eN in enumerate(presentElectrodes):
    electrodeErrorBarStyles[eN] = dict(
        marker=allFilledMarkers[eIdx],
        elinewidth=sns.plotting_context()['patch.linewidth'],
        capsize=sns.plotting_context()['xtick.major.size'])
electrodeErrorBarStyleDF = pd.DataFrame(electrodeErrorBarStyles)
keepCols = [
    'segment', 'originalIndex', 't',
    'feature', 'freqBandName', 'lag',
    'stimCondition', 'kinematicCondition'] + stimulusConditionNames
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
print('Plotting select features:')
print(', '.join(["'{}#0'".format(fN) for fN in keepColsForPlot]))
plotRCPieces = plotRC.loc[plotRC['feature'].isin(keepColsForPlot), :].copy()
######
refGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] == 'NA_0.0', :]
testGroupPieces = plotRCPieces.loc[plotRCPieces['stimCondition'] != 'NA_0.0', :]
#
refGroup = plotRC.loc[plotRC['stimCondition'] == 'NA_0.0', :]
testGroup = plotRC.loc[plotRC['stimCondition'] != 'NA_0.0', :]

def genStatsAnnotator(ampDF, relDF, hN, hP):
    def statsAnnotator(g, ro, co, hu, dataSubset):
        emptySubset = (
            (dataSubset.empty) or
            (dataSubset.iloc[:, 0].isna().all()))
        if not emptySubset:
            if not hasattr(g.axes[ro, co], 'starsAnnotated'):
                xLim = g.axes[ro, co].get_xlim()
                yLim = g.axes[ro, co].get_ylim()
                trans = transforms.blended_transform_factory(
                    g.axes[ro, co].transAxes, g.axes[ro, co].transData)
                # dx = (xLim[1] - xLim[0]) / 5
                # dy = (yLim[1] - yLim[0]) / 5
                for hn, group in dataSubset.groupby([hN]):
                    rn = group[g._row_var].unique()[0]
                    cn = group[g._col_var].unique()[0]
                    thisElectrode = compoundAnnLookupDF.loc[cn, 'electrode']
                    st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialAmplitude', level='names')
                    x = group[g._x_var].max()
                    y = group.groupby(g._x_var).mean().loc[x, g._y_var]
                    message = ''
                    if st['reject'].iloc[0]:
                        message += '*'
                    st = ampDF.xs(rn, level=g._row_var).xs(thisElectrode, level='electrode').xs(hn, level=hN).xs('trialRateInHz', level='names')
                    if st['reject'].iloc[0]:
                        message += '^'
                    rst = relDF.xs(rn, level=g._row_var).xs(cn, level=g._col_var).xs(hn, level=hN)
                    # if rst['pval'].iloc[0] < 1 - confidence_alpha:
                    if rst['reject'].iloc[0]:
                        message += '+'
                    if len(message):
                        g.axes[ro, co].text(1., y, message, color=hP[hn], va='bottom', ha='left', transform=trans)
                if (ro == 0) and (co == 0):
                    g.axes[ro, co].text(
                        0.95, 0.95,
                        '\n'.join([
                            '+: highest amp. stim. vs baseline (pval < 1e-3)',
                            '*: amplitude vs auc (pval < 1e-3)',
                            '^: rate vs auc (pval < 1e-3)']),
                        va='top', ha='right', transform=g.axes[ro, co].transAxes)
                g.axes[ro, co].starsAnnotated = True
        return
    return statsAnnotator

def genNumSigAnnotator(pvDF, xOrder=None, hueVar=None, palette=None, fontOpts={}, width=0.9, nudgeFactor=1.2):
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
                        message = '{}/{}/{}'.format(int(thisEntry['under']), int(thisEntry['ns']), int(thisEntry['over']))
                        x = xIdx + offsets[hIdx]
                        y = 0
                        g.axes[ro, co].text(
                            x, y, message,
                            transform=trans, color=huePalette[hLabel],
                            **fontOpts)
            g.axes[ro, co].pvalsAnnotated = True
        else:
            print('ro {} co {} g.axes[ro, co].pvalsAnnotated {}'.format(ro, co, g.axes[ro, co].pvalsAnnotated))
        return
    return numSigAnnotator

def genNumSigAnnotatorV2(
        pvDF, xOrder=None, hueVar=None,
        palette=None, fontOpts={}, width=0.9, nudgeFactor=1.2):
    def numSigAnnotator(g, theAx, rowVar, colVar):
        if not hasattr(theAx, 'pvalsAnnotated'):
            axesNeedNudging = (not hasattr(g, 'nudgedAxesForPValAnnotation')) or (not g._sharey)
            if axesNeedNudging:
                yLim = theAx.get_ylim()
                yExt = yLim[1] - yLim[0]
                theAx.set_ylim([yLim[1] - nudgeFactor * yExt, yLim[1]])
                print('numSigAnnotator: extended ylims to make room')
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
                            print('Warning!\n{}'.format(thisEntry.loc[['under', 'ns', 'over']]))
                            continue
                        message = '{}/{}/{}'.format(
                            int(thisEntry['under']), int(thisEntry['ns']), int(thisEntry['over']))
                        x = xIdx + offsets[hIdx]
                        y = 0
                        theAx.text(
                            x, y, message,
                            transform=trans, color=huePalette[hLabel],
                            **fontOpts)
                        print('numSigAnnotator: annotating axes')
                    else:
                        print('Warning! No matches for\n{} == {} (columns)\n{} == {}(row)\n{} == {} (hue)'.format(g._col_var, colVar, g._row_var, rowVar, hueVar, hLabel))
            theAx.pvalsAnnotated = True
        else:
            print('numSigAnnotator: axes already annotated')
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
        errorBarOptsDict=None
        ):
    semanticNames = [n for n in [row, col, x, hue, style] if n is not None]
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
        if jitter:
            panelGroups = [n for n in [row, col, hue] if n is not None]
            xJitters = pd.Series(0, index=fullDataset.index)
            if len(panelGroups):
                grouper = fullDataset.sort_values(semanticNames).groupby(panelGroups)
            else:
                grouper = ('all', fullDataset.sort_values(semanticNames))
            for name, group in grouper:
                rawJitters = 0.2 * rng.random(group.shape[0]) + 0.8
                rawJitters = np.cumsum(rawJitters) / np.sum(rawJitters)
                # rng.shuffle(rawJitters)
                # pdb.set_trace()
                xJitters.loc[group.index] = (rawJitters.copy() - 0.5) * width / nHues
            #
            def xPosFun(dfRow):
                if dfRow[hue] in xHueMappingDict:
                    if dfRow[x] in xHueMappingDict[dfRow[hue]].index:
                        return xHueMappingDict[dfRow[hue]][dfRow[x]] + xJitters.loc[dfRow.name]
                else:
                    return np.nan
            allXPos = fullDataset.apply(xPosFun, axis='columns')
        else:
            def xPosFun(dfRow):
                if dfRow[hue] in xHueMappingDict:
                    if dfRow[x] in xHueMappingDict[dfRow[hue]]:
                        return xHueMappingDict[dfRow[hue]][dfRow[x]]
                else:
                    return np.nan
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
                    mSize = 2 * thisStyle['markersize'] if ('markersize' in thisStyle) else 2 * sns.plotting_context()['lines.markersize']
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

print('Saving plots to {}'.format(pdfPath))
with PdfPages(pdfPath) as pdf:
    # with contextlib.nullcontext() as pdf:
    rowVar = 'feature'
    rowOrder = sorted(np.unique(plotRC[rowVar]))
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
    figHeight = 2. + 2. * len(rowOrder)
    figWidth = 2. + len(colOrder) * len(freqBandOrderExtended)
    if arguments['plotThePieces']:
        ####
        widthRatios = [3] * np.unique(testGroupPieces[colVar]).shape[0] + [1]
        plotLimsMin = plotRCPieces.groupby(rowVar).min()[whichRAUC]
        plotLimsMax = plotRCPieces.groupby(rowVar).max()[whichRAUC]
        plotLimsRange = plotLimsMax - plotLimsMin
        plotLimsMin -= plotLimsRange / 100
        plotLimsMax += plotLimsRange / 100
        xJ = testGroupPieces[amplitudeFieldName].diff().dropna().abs().unique().mean() / 20
        height = figHeight / testGroupPieces[rowVar].nunique()
        width = figWidth / testGroupPieces[colVar].nunique()
        aspect = width / height
        g = sns.lmplot(
            col=colVar, col_order=colOrder,
            row=rowVar,
            x=amplitudeFieldName, y=whichRAUC,
            hue=hueName, hue_order=hueOrder, palette=huePalette,
            data=testGroupPieces,
            ci=95, n_boot=100,
            x_jitter=xJ,
            scatter_kws=dict(s=2.5),
            height=height, aspect=aspect,
            facet_kws=dict(
                sharey=False, sharex=False, margin_titles=True,
                gridspec_kws=dict(width_ratios=widthRatios)),
            )
        plotProcFuns = [
            genStatsAnnotator(ampStatsDF, relativeStatsDF, hueName, huePalette),
            asp.genTitleChanger(prettyNameLookup)]
        for (ro, co, hu), dataSubset in g.facet_data():
            if len(plotProcFuns):
                for procFun in plotProcFuns:
                    procFun(g, ro, co, hu, dataSubset)
        for (row_val, col_val), ax in g.axes_dict.items():
            if col_val == 'NA_0.0':
                refMask = (refGroupPieces[rowVar] == row_val)
                if refMask.any():
                    refData = refGroupPieces.loc[refMask, :]
                else:
                    refData = refGroupPieces
                sns.boxplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC,
                    hue=hueName, hue_order=hueOrder, palette=huePaletteAlpha,
                    data=refData, saturation=0.,
                    ax=ax, whis=np.inf, dodge=True)
                sns.stripplot(
                    x=hueName, order=hueOrder,
                    y=whichRAUC, palette=huePalette,
                    hue=hueName, hue_order=hueOrder, data=refData,
                    ax=ax, size=sns.plotting_context['lines.markersize'], dodge=True)
                ax.set_xlabel('')
                ax.set_xticks([])
                if ax.get_legend() is not None:
                    ax.get_legend().remove()
        g.set_axis_labels('Stimulation amplitude (uA)', 'Normalized AUC')
        asp.reformatFacetGridLegend(
            g, titleOverrides={
                'kinematicCondition': 'Movement type'
            },
            contentOverrides={
                'NA_NA': 'No movement',
                'CW_outbound': 'Start of movement (extension)',
                'CW_return': 'Return to start (flexion)'
            },
            styleOpts=styleOpts)
        g.resize_legend(adjust_subtitles=True)
        for (rN, cN), ax in g.axes_dict.items():
            ax.set_ylim([plotLimsMin.loc[rN], plotLimsMax.loc[rN]])
        # g.axes[0, 0].set_ylim(plotLims)
        g.tight_layout(pad=styleOpts['tight_layout.pad'])
        if not consoleDebug:
            pdf.savefig(bbox_inches='tight', pad_inches=0,)
        if arguments['showFigures']:
            plt.show()
        else:
            plt.close()
    if True:
        stripplotKWArgs = dict(
            dodge=True, size=sns.plotting_context()['lines.markersize'],
            rasterized=False,
            linewidth=sns.plotting_context()['patch.linewidth'] / 2
            )
        stripplotKWArgsMahal = stripplotKWArgs.copy()
        stripplotKWArgsMahal['size'] = 1.5 * sns.plotting_context()['lines.markersize']
        stripplotKWArgsMahal['jitter'] = False
        boxplotKWArgs = dict(
            dodge=True,
            whis=np.inf,
            linewidth=sns.plotting_context()['lines.linewidth'],
            saturation=0.3,
            )
        colVar = 'electrode'
        rowVar = 'kinematicCondition'
        ################################################################################################################
        plotAmpStatsDF = ampStatsDF.reset_index()
        plotAmpStatsDF.loc[:, 'electrode'] = plotAmpStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotAmpStatsDF.loc[:, 'namesAndMD'] = plotAmpStatsDF['names']
        plotAmpStatsDF.loc[plotAmpStatsDF['isMahalDist'], 'namesAndMD'] += '_md'
        ################################################################################################################
        dummyEntriesReject = plotAmpStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'namesAndMD', 'kinematicCondition']).copy()
        dummyEntriesReject.loc[:, ['coef', 'coefStd', 'se', 'T', 'pval', 'r2', 'adj_r2', 'relimp', 'relimp_perc']] = np.nan
        dummyEntriesReject.loc[:, 'reject'] = True
        dummyEntriesDoNotReject = dummyEntriesReject.copy()
        dummyEntriesDoNotReject.loc[:, 'reject'] = False
        plotAmpStatsDF = pd.concat([plotAmpStatsDF, dummyEntriesReject, dummyEntriesDoNotReject], ignore_index=True)
        #######
        thisMaskAmp = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) & plotAmpStatsDF['electrode'].isin(lOfElectrodesToPlot) & ~plotAmpStatsDF['isMahalDist']
        if thisMaskAmp.any():
            thisMaskAmpMahal = plotAmpStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot) & plotAmpStatsDF['electrode'].isin(lOfElectrodesToPlot) & plotAmpStatsDF['isMahalDist']
            thisFreqBandOrder = [
                fN for fN in freqBandOrderExtended if fN in plotAmpStatsDF.loc[thisMaskAmp, 'freqBandName'].unique().tolist()]

            rowOrder = [rN for rN in lOfKinematicsToPlot if rN in plotAmpStatsDF['kinematicCondition'].to_list()]
            #
            thisPalette = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF.loc[thisMaskAmp, 'namesAndMD'])]
            thisPaletteMahal = allAmpPalette.loc[allAmpPalette.index.isin(plotAmpStatsDF.loc[thisMaskAmpMahal, 'namesAndMD'])]
            colOrder = [eN for eN in plotAmpStatsDF[colVar].unique().tolist() if ((eN !='NA') and (eN in lOfElectrodesToPlot))]
            #pdb.set_trace()
            if thisPalette.shape[0] > 1:
                stripJitter = 0.99 / (thisPalette.shape[0] + 1)
            else:
                stripJitter = 0.99
            # recruitment regression coeffs
            figHeight = 2 + 2. * len(rowOrder)
            figWidth = 2 + len(colOrder) * len(freqBandOrderExtended)
            height = figHeight / plotAmpStatsDF.loc[thisMaskAmp, rowVar].nunique()
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
                g = sns.catplot(
                    y=yVar,
                    x='freqBandName', order=thisFreqBandOrder,
                    col=colVar, col_order=colOrder,
                    row=rowVar, row_order=rowOrder,
                    hue='namesAndMD', hue_order=thisPalette.index.to_list(),
                    palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(), color='w',
                    data=plotAmpStatsDF.loc[thisMaskAmp, :],  # & plotAmpStatsDF['reject']
                    height=height, aspect=aspect,
                    sharey=True, sharex=True,  margin_titles=True,
                    kind='box', **boxplotKWArgs
                    )
                for name, ax in g.axes_dict.items():
                    rowName, colName = name
                    # non-significant are transparent
                    subSetMask = thisMaskAmp & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & (~plotAmpStatsDF['reject']) #& plotAmpStatsDF[yVar].notna()
                    if subSetMask.any():
                        if (not plotAmpStatsDF.loc[subSetMask, :].dropna().empty):
                            sns.stripplot(
                                data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                                y=yVar, x='freqBandName',
                                order=thisFreqBandOrder,
                                hue='namesAndMD', hue_order=thisPalette.index.to_list(),
                                palette=thisPalette.to_dict(),
                                jitter=stripJitter,
                                alpha=0.3, edgecolor='none', **stripplotKWArgs)
                    # significant are opaque
                    subSetMask = thisMaskAmp & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & plotAmpStatsDF['reject'] # & plotAmpStatsDF[yVar].notna()
                    if subSetMask.any():
                        if (not plotAmpStatsDF.loc[subSetMask, :].dropna().empty):
                            sns.stripplot(
                                data=plotAmpStatsDF.loc[subSetMask, :], ax=ax,
                                y=yVar, x='freqBandName',
                                order=thisFreqBandOrder, hue='namesAndMD',
                                hue_order=thisPalette.index.to_list(),
                                palette=thisPalette.to_dict(), edgecolor='k',
                                jitter=stripJitter,
                                **stripplotKWArgs
                                )
                    addMahalGsHere = True
                    # non-significant are transparent (mahal)
                    subSetMaskMahal = thisMaskAmpMahal & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & (~plotAmpStatsDF['reject'])# & plotAmpStatsDF[yVar].notna()
                    if subSetMaskMahal.any() and addMahalGsHere:
                        if (not plotAmpStatsDF.loc[subSetMaskMahal, :].dropna().empty):
                            sns.stripplot(
                                data=plotAmpStatsDF.loc[subSetMaskMahal, :], ax=ax,
                                y=yVar, x='freqBandName',
                                order=thisFreqBandOrder,
                                hue='namesAndMD', hue_order=thisPaletteMahal.index.to_list(), palette=thisPaletteMahal.to_dict(),
                                alpha=0.5, edgecolor='none', **stripplotKWArgsMahal)
                    # significant are opaque (mahal)
                    subSetMaskMahal = thisMaskAmpMahal & (plotAmpStatsDF[rowVar] == rowName) & (plotAmpStatsDF[colVar] == colName) & plotAmpStatsDF['reject'] # & plotAmpStatsDF[yVar].notna()
                    if subSetMaskMahal.any() and addMahalGsHere:
                        if (not plotAmpStatsDF.loc[subSetMaskMahal, :].dropna().empty):
                            sns.stripplot(
                                data=plotAmpStatsDF.loc[subSetMaskMahal, :], ax=ax,
                                y=yVar, x='freqBandName',
                                order=thisFreqBandOrder, hue='namesAndMD',
                                hue_order=thisPaletteMahal.index.to_list(), palette=thisPaletteMahal.to_dict(),
                                edgecolor='k',
                                **stripplotKWArgsMahal
                                )
                    # ax.tick_params(axis='x', labelrotation=30)
                    xTickLabels = ax.get_xticklabels()
                    if len(xTickLabels):
                        newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                        ax.set_xticklabels(
                            newXTickLabels, rotation=30, va='top', ha='right')
                    ax.axhline(0, c='r', zorder=2.5)
                    for xJ in range(1, len(thisFreqBandOrder), 2):
                        ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                    if ax.get_legend() is not None:
                        ax.get_legend().remove()
                enableNumSigAnnotator = False
                plotProcFuns = [asp.genTitleChanger(prettyNameLookup)]
                if enableNumSigAnnotator:
                    plotProcFuns += [
                    genNumSigAnnotator(
                        pvalDF.reset_index(),
                        xOrder=thisFreqBandOrder, hueVar='namesAndMD', palette=thisPalette,
                        fontOpts=dict(
                            va='top', ha='right',
                            fontsize=sns.plotting_context()["font.size"],
                            fontweight='bold', rotation=30))]
                for (ro, co, hu), dataSubset in g.facet_data():
                    if len(plotProcFuns):
                        for procFun in plotProcFuns:
                            procFun(g, ro, co, hu, dataSubset)
                g.suptitle('Coefficient distribution for AUC regression')
                asp.reformatFacetGridLegend(
                    g, titleOverrides={
                        'namesAndMD': 'Regressors'
                    },
                    contentOverrides={
                        'namesAndMD': 'Regressors',
                        'trialAmplitude': 'Stimulation amplitude',
                        'trialAmplitude:trialRateInHz': 'Stimulation rate interaction',
                        'trialAmplitude_md': 'Stimulation amplitude (Mahal. dist.)',
                        'trialAmplitude:trialRateInHz_md': 'Stimulation rate interaction (Mahal. dist.)',
                    },
                styleOpts=styleOpts)
                g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
                g.resize_legend(adjust_subtitles=True)
                g.tight_layout(pad=styleOpts['tight_layout.pad'])
                ########################
                if not consoleDebug:
                    pdf.savefig(bbox_inches='tight', pad_inches=0,)
                if arguments['showFigures']:
                    plt.show()
                else:
                    plt.close()
            ######
        plotRelativeStatsDF = relativeStatsDF.reset_index()
        plotStatsList = [plotRelativeStatsDF]
        for cN in plotRelativeStatsDF['kinematicCondition'].unique():
            if cN == 'NA_NA':
                continue
            noStimMask = (relativeStatsNoStimDF['A'] == cN) & (relativeStatsNoStimDF['B'] == 'NA_NA')
            if noStimMask.any():
                takeThese = relativeStatsNoStimDF.loc[noStimMask, :].reset_index()
                takeThese.loc[:, 'kinematicCondition'] = cN
                takeThese.loc[:, 'kinAndElecCondition'] = 'NA_{}'.format(cN)
                takeThese.loc[:, 'stimCondition'] = 'NA_0.0'
                takeThese.loc[:, 'electrode'] = 'NA'
                takeThese.loc[:, 'trialRateInHz'] = 0.
                # drop duplicates to avoid having multiple mahalanobis distances?
                # takeThese.drop_duplicates(subset=['feature'], inplace=True)
                relevantColumns = [cN for cN in plotRelativeStatsDF.columns if cN in takeThese.columns]
                try:
                    plotStatsList.append(takeThese.loc[:, relevantColumns].copy())
                except Exception:
                    traceback.print_exc()
        plotRelativeStatsDF = pd.concat(plotStatsList)
        plotRelativeStatsDF.loc[:, 'trialRateInHzStr'] = plotRelativeStatsDF['trialRateInHz'].apply(lambda x: '{}'.format(x))
        plotRelativeStatsDF.loc[:, 'electrode'] = plotRelativeStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotRelativeStatsDF.loc[plotRelativeStatsDF['isMahalDist'], 'trialRateInHzStr'] += '_md'
        #####
        dummyEntriesReject = plotRelativeStatsDF.drop_duplicates(subset=['freqBandName', 'electrode', 'trialRateInHz', 'isMahalDist', 'kinematicCondition']).copy()
        dummyEntriesReject.loc[:, ['hedges', 'T', 'pval', 'cohen-d']] = np.nan
        dummyEntriesReject.loc[:, 'reject'] = True
        dummyEntriesDoNotReject = dummyEntriesReject.copy()
        dummyEntriesDoNotReject.loc[:, 'reject'] = False
        plotRelativeStatsDF = pd.concat([plotRelativeStatsDF, dummyEntriesReject, dummyEntriesDoNotReject], ignore_index=True)
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(x['feature'].replace('#0', '')), axis='columns')
        ###
        thisMaskRel = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & (plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot)) & ~plotRelativeStatsDF['isMahalDist']
        thisMaskRelStimOnly = thisMaskRel & (plotRelativeStatsDF['stimCondition'] != 'NA_0.0')
        thisMaskRelNoStim = thisMaskRel & (plotRelativeStatsDF['stimCondition'] == 'NA_0.0')
        thisMaskRelMahal = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & (plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot)) & plotRelativeStatsDF['isMahalDist']
        thisMaskRelStimOnlyMahal = thisMaskRelMahal & (plotRelativeStatsDF['stimCondition'] != 'NA_0.0')
        thisMaskRelNoStimMahal = thisMaskRelMahal & (plotRelativeStatsDF['stimCondition'] == 'NA_0.0')
        thisFreqBandOrder = [
            fN
            for fN in freqBandOrderExtended
            if fN in plotRelativeStatsDF.loc[thisMaskRel, 'freqBandName'].unique().tolist()]
        ####
        thisFullPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[(plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot)), 'trialRateInHzStr'])]
        thisStimPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelStimOnly, 'trialRateInHzStr'])]
        thisNoStimPalette = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelNoStim, 'trialRateInHzStr'])]
        ######
        thisFullPaletteMahal = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[(plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot)), 'trialRateInHzStr'])]
        thisStimPaletteMahal = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelStimOnlyMahal, 'trialRateInHzStr'])]
        thisNoStimPaletteMahal = allRelPalette.loc[allRelPalette.index.isin(plotRelativeStatsDF.loc[thisMaskRelNoStimMahal, 'trialRateInHzStr'])]
        # thisPalette = pd.Series(sns.color_palette('Set2_r', 4), index=['50.0', '50.0_md', '100.0', '100.0_md'])
        colOrder = ['NA'] + [eN for eN in plotRelativeStatsDF[colVar].sort_values().unique().tolist() if ((eN !='NA') and (eN in lOfElectrodesToPlot))]
        #
        ## countDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(['kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).count()['reject']
        ## passDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(['kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).sum()['reject']
        ## pvalDF = pd.concat([countDF, passDF], axis='columns')
        ## pvalDF.columns = ['count', 'pass']
        #
        def countOverUnder(df):
            res = pd.Series({
                'over': int(((df['T'] > 0) & df['reject']).sum()),
                'ns': int((~df['reject']).sum()),
                'under': int(((df['T'] < 0) & df['reject']).sum()),
            })
            return res
        #
        pvalDF = plotRelativeStatsDF.loc[thisMaskRel, :].dropna().groupby(['kinematicCondition', colVar, 'freqBandName', 'trialRateInHzStr']).apply(countOverUnder).fillna(0)
        widthRatios = [3] * np.unique(plotRelativeStatsDF[colVar]).shape[0]
        widthRatios[0] = 2
        # plotRelativeStatsDF.loc[plotRelativeStatsDF['electrode'] == 'NA', :]
        ### effect size catplot stripplot
        figHeight = 2 + 2. * len(rowOrder)
        figWidth = 2 + (len(colOrder) + 1) * len(freqBandOrderExtended)
        height = figHeight / plotRelativeStatsDF.loc[thisMaskRel, rowVar].nunique()
        width = figWidth / plotRelativeStatsDF.loc[thisMaskRel, colVar].nunique()
        aspect = width / height
        for yVar in ['hedges']:
            g = sns.catplot(
                y=yVar,
                x='freqBandName',
                order=thisFreqBandOrder,
                col=colVar, col_order=colOrder,
                row=rowVar, row_order=rowOrder,
                height=height, aspect=aspect,
                sharey=True, sharex=True, margin_titles=True,
                hue='trialRateInHzStr', hue_order=thisStimPalette.index.to_list(),
                palette=thisStimPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                data=plotRelativeStatsDF.loc[thisMaskRelStimOnly, :],  # & plotRelativeStatsDF['reject']
                facet_kws=dict(
                    gridspec_kws=dict(width_ratios=widthRatios)),
                kind='box',
                # kind='violin', inner=None, cut=1, width=0.9,
                # saturation=0.2,
                **boxplotKWArgs
                )
            for name, ax in g.axes_dict.items():
                rowName, colName = name
                if colName == 'NA':
                    # movement
                    thisPalette = thisNoStimPalette
                    expandDays = True
                    if expandDays:
                        stripHue = 'expName'
                        presentExps = plotRelativeStatsDF['expName'].unique().tolist()
                        stripHueOrder = presentExps
                        stripHueOrderMahal = presentExps
                        print('presentExps = {}'.format(presentExps))
                        stripJitter = thisStimPalette.shape[0] / (len(stripHueOrder) + 1)
                        stripPalette = {expN: sns.utils.alter_color(allRelPalette['0.0'], h=-0.01 * expIdx, s=0.1 * expIdx, l=0.1 * expIdx) for expIdx, expN in enumerate(presentExps)}
                        stripPaletteMahal = {expN: sns.utils.alter_color(allRelPalette['0.0_md'], h=-0.01 * expIdx, s=0.1 * expIdx, l=0.1 * expIdx) for expIdx, expN in enumerate(presentExps)}
                    else:
                        stripHue = 'trialRateInHzStr'
                        stripHueOrder = thisNoStimPalette.index.to_list()
                        stripPalette = thisNoStimPalette.to_dict()
                        stripPaletteMahal = thisNoStimPaletteMahal.to_dict()
                        stripHueOrderMahal = thisNoStimPaletteMahal.index.to_list()
                        stripJitter = thisStimPalette.shape[0] / (len(stripHueOrder) + 1)
                    subSetMask = (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) & plotRelativeStatsDF[yVar].notna()
                    if subSetMask.any():
                        #could turn off as it isn't very visible anyway
                        sns.boxplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                            palette=thisPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
                            **boxplotKWArgs)
                else:
                    # stim
                    thisPalette = thisStimPalette
                    # thisPaletteMahal = thisStimPaletteMahal
                    stripHue = 'trialRateInHzStr'
                    stripHueOrder = thisStimPalette.index.to_list()
                    stripPalette = thisStimPalette.to_dict()
                    stripPaletteMahal = thisStimPaletteMahal.to_dict()
                    stripHueOrderMahal = thisStimPaletteMahal.index.to_list()
                    stripJitter = 1. / (len(stripHueOrder) + 1)
                # plot non-significant observations with transparency
                subSetMask = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        (~plotRelativeStatsDF['reject']) & plotRelativeStatsDF[yVar].notna() & thisMaskRel)
                if subSetMask.any():
                    if not (plotRelativeStatsDF.loc[subSetMask, :].dropna().empty):
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrder, palette=stripPalette,
                            jitter=stripJitter,
                            alpha=0.3, edgecolor='none', **stripplotKWArgs)
                # plot significant observations fully opaque
                subSetMask = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        plotRelativeStatsDF['reject'] & plotRelativeStatsDF[yVar].notna() & thisMaskRel)
                if subSetMask.any():
                    if not (plotRelativeStatsDF.loc[subSetMask, :].dropna().empty):
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMask, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrder, palette=stripPalette,
                            jitter=stripJitter,
                            edgecolor = 'k',
                            **stripplotKWArgs
                            )
                #####
                # plot non-significant observations with transparency (mahal)
                subSetMaskMahal = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        (~plotRelativeStatsDF['reject']) & plotRelativeStatsDF[yVar].notna() & thisMaskRelMahal)
                if subSetMaskMahal.any() and addMahalGsHere:
                    if not (plotRelativeStatsDF.loc[subSetMaskMahal, :].dropna().empty):
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrderMahal, palette=stripPaletteMahal,
                            alpha=0.5, edgecolor='none', **stripplotKWArgsMahal)
                # plot significant observations fully opaque (mahal)
                subSetMaskMahal = (
                        (plotRelativeStatsDF[rowVar] == rowName) & (plotRelativeStatsDF[colVar] == colName) &
                        plotRelativeStatsDF['reject'] & plotRelativeStatsDF[yVar].notna() & thisMaskRelMahal)
                if subSetMaskMahal.any() and addMahalGsHere:
                    if not (plotRelativeStatsDF.loc[subSetMaskMahal, :].dropna().empty):
                        sns.stripplot(
                            data=plotRelativeStatsDF.loc[subSetMaskMahal, :], ax=ax,
                            y=yVar, x='freqBandName', order=thisFreqBandOrder,
                            hue=stripHue, hue_order=stripHueOrderMahal, palette=stripPaletteMahal,
                            edgecolor = 'k',
                            **stripplotKWArgsMahal
                            )
                nSigAnnotator = genNumSigAnnotatorV2(
                    pvalDF.reset_index(),
                    xOrder=thisFreqBandOrder,
                    hueVar='trialRateInHzStr', palette=thisPalette,
                    fontOpts=dict(
                        va='bottom', ha='center',
                        fontweight='bold', rotation=45))
                if enableNumSigAnnotator:
                    nSigAnnotator(g, ax, rowName, colName)
                ax.axhline(0, c='r', zorder=2.5)
                # ax.tick_params(axis='x', labelrotation=30)
                xTickLabels = ax.get_xticklabels()
                if len(xTickLabels):
                    newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                    ax.set_xticklabels(newXTickLabels, rotation=30, va='top', ha='right')
                for xJ in range(1, len(thisFreqBandOrder), 2):
                    ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                if (ax.get_legend() is not None):
                    ax.get_legend().remove()
                ax.set_xlim([-0.5, len(thisFreqBandOrder) - 0.5])
            plotProcFuns = [
                asp.genTitleChanger(prettyNameLookup),
                ]
            for (ro, co, hu), dataSubset in g.facet_data():
                if len(plotProcFuns):
                    for procFun in plotProcFuns:
                        procFun(g, ro, co, hu, dataSubset)
            g.suptitle('Effect size distribution for stim vs no-stim comparisons')
            asp.reformatFacetGridLegend(
                g, titleOverrides=prettyNameLookup,
                contentOverrides=prettyNameLookup,
            styleOpts=styleOpts)
            g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
            g.resize_legend(adjust_subtitles=True)
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            if not consoleDebug:
                pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            plotMahalDistSummary = True
            ######## mahal dist plots
            if plotMahalDistSummary:
                thisMaskRel = (plotRelativeStatsDF['kinematicCondition'].isin(lOfKinematicsToPlot)) & (plotRelativeStatsDF['electrode'].isin(lOfElectrodesToPlot)) & plotRelativeStatsDF['isMahalDist']
                figHeight = 2 + 2. * plotRelativeStatsDF.loc[thisMaskRel, rowVar].nunique()
                figWidth = 1. + 0.5 * len(freqBandOrderExtended) * len(lOfElectrodesToPlot)
                height = figHeight / plotRelativeStatsDF.loc[thisMaskRel, rowVar].nunique()
                width = figWidth
                aspect = width / height
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
                        sharey=True, sharex=True, margin_titles=True,
                        hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                        palette=thisPalette.to_dict(),
                        data=plotRelativeStatsDF.loc[thisMaskRel, :])
                    legendData, mDSummaryPlotFun = genMahalSummaryPlotFun(
                        fullDataset=plotRelativeStatsDF.loc[thisMaskRel, :],
                        x='freqBandName', order=thisFreqBandOrder, y=yVar,
                        row=rowVar, row_order=rowOrder,
                        hue='trialRateInHzStr', hue_order=thisPalette.index.to_list(),
                        style='electrode', style_palette=electrodeErrorBarStyleDF,
                        yerr='cohen_err', error_palette=thisErrPalette.to_dict(),
                        palette=thisPalette.to_dict(),
                        errorBarOptsDict=dict(
                            markeredgecolor='k',
                            markeredgewidth=sns.plotting_context()['patch.linewidth'],
                            markersize=sns.plotting_context()['lines.markersize'],
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
                        ax.axhline(0, c='.1', zorder=2.5)
                        # ax.tick_params(axis='x', labelrotation=30)
                        xTickLabels = ax.get_xticklabels()
                        if len(xTickLabels):
                            newXTickLabels = [applyPrettyNameLookup(tL.get_text()) for tL in xTickLabels]
                            ax.set_xticklabels(newXTickLabels, rotation=30, va='top', ha='right')
                        for xJ in range(1, len(thisFreqBandOrder), 2):
                            # for xJ in range(0, plotRelativeStatsDF[colVar].unique().shape[0], 2):
                            ax.axvspan(-0.45 + xJ, 0.45 + xJ, color="0.1", alpha=0.1, zorder=1.)
                        if ax.get_legend() is not None:
                            ax.get_legend().remove()
                    plotProcFuns = [asp.genTitleChanger(prettyNameLookup),]
                    for (ro, co, hu), dataSubset in g.facet_data():
                        if len(plotProcFuns):
                            for procFun in plotProcFuns:
                                procFun(g, ro, co, hu, dataSubset)
                    g.suptitle('Effect size distribution for stim vs no-stim comparisons (Mahal dist)')
                    g.add_legend(legend_data=legendData, label_order=list(legendData.keys()))
                    asp.reformatFacetGridLegend(
                        g, titleOverrides=prettyNameLookup,
                        contentOverrides=prettyNameLookup,
                    styleOpts=styleOpts)
                    g.set_axis_labels(prettyNameLookup['freqBandName'], prettyNameLookup[yVar])
                    g.resize_legend(adjust_subtitles=True)
                    g.tight_layout(pad=styleOpts['tight_layout.pad'])
                    if not consoleDebug:
                        pdf.savefig(bbox_inches='tight', pad_inches=0,)
                    if arguments['showFigures']:
                        plt.show()
                    else:
                        plt.close()
        #####################################################################
        #####################################################################
        # dummy plot for legends
        g = sns.catplot(
            y=yVar,
            x=colVar, order=colOrder,
            height=height, aspect=aspect,
            sharey=True, sharex=True, margin_titles=True,
            hue='trialRateInHzStr', hue_order=thisFullPalette.index.to_list(),
            palette=thisFullPalette.apply(sns.utils.alter_color, l=0.5).to_dict(),
            data=plotRelativeStatsDF, # & plotRelativeStatsDF['reject']
            kind='box',
            # kind='violin', inner=None, cut=1, width=0.9,
            # saturation=0.2,
            **boxplotKWArgs
            )
        g.suptitle('Dummy plot to get full legend from')
        legendData = {}
        legendData['coefStd'] = mpl.patches.Patch(alpha=0, linewidth=0)
        for factorName, factorColor in allAmpPalette.items():
            if '_md' in factorName:
                legendData[factorName] = mpl.lines.Line2D(
                    [0], [0], marker='o',
                    markerfacecolor=factorColor, markeredgecolor='k',
                    markeredgewidth=sns.plotting_context()['patch.linewidth'],
                    markersize=sns.plotting_context()['lines.markersize'],
                    linestyle='none')
            else:
                legendData[factorName] = mpl.lines.Line2D(
                    [0], [0], marker='o',
                    markerfacecolor=factorColor, markeredgecolor='k',
                    markeredgewidth=sns.plotting_context()['patch.linewidth'],
                    markersize=sns.plotting_context()['lines.markersize'],
                    linestyle='none')
        legendData['trialRateInHzStr'] = mpl.patches.Patch(alpha=0, linewidth=0)
        for rateName, rateColor in allRelPalette.items():
            if '_md' in rateName:
                legendData[rateName] = mpl.lines.Line2D(
                    [0], [0], marker='o',
                    markerfacecolor=rateColor, markeredgecolor='k',
                    markeredgewidth=sns.plotting_context()['patch.linewidth'],
                    markersize=1.5 * sns.plotting_context()['lines.markersize'],
                    linestyle='none')
            else:
                legendData[rateName] = mpl.lines.Line2D(
                    [0], [0], marker='o',
                    markerfacecolor=rateColor, markeredgecolor='k',
                    markeredgewidth=sns.plotting_context()['patch.linewidth'],
                    markersize=sns.plotting_context()['lines.markersize'],
                    linestyle='none')
        legendData['p-value significance'] = mpl.patches.Patch(alpha=0, linewidth=0)
        legendData['p < {:.2f}'.format(confidence_alpha)] = mpl.lines.Line2D(
            [0], [0], marker='o',
            markerfacecolor='darkgray', markeredgecolor='k',
            markeredgewidth=sns.plotting_context()['patch.linewidth'],
            markersize=sns.plotting_context()['lines.markersize'],
            linestyle='none')
        legendData['p > {:.2f} (n.s.)'.format(confidence_alpha)] = mpl.lines.Line2D(
            [0], [0], marker='o',
            markerfacecolor='lightgray', markeredgecolor='k',
            markeredgewidth=sns.plotting_context()['patch.linewidth'],
            markersize=sns.plotting_context()['lines.markersize'],
            linestyle='none')
        g.add_legend(
            legend_data=legendData, label_order=list(legendData.keys()), title='dummy legends')
        asp.reformatFacetGridLegend(
            g, titleOverrides=prettyNameLookup,
            contentOverrides=prettyNameLookup,
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
    hasTopo = relativeStatsDF.groupby(['xCoords', 'yCoords']).ngroups > 1
    plotTDist = False
    if hasTopo: # histogram t stats
        rowVar = 'feature'
        rowOrder = sorted(np.unique(plotRC[rowVar]))
        colVar = 'stimCondition'
        colOrder = sorted(np.unique(plotRC[colVar]))
        colWrap = min(3, len(colOrder))
        hueName = 'kinematicCondition'
        # hueOrder = sorted(np.unique(plotRC[hueName]))
        hueOrder = ['NA_NA', 'CW_outbound', 'CW_return']
        pal = sns.color_palette("Set2")
        huePalette = {hN: pal[hIdx] for hIdx, hN in enumerate(hueOrder)}
        huePaletteAlpha = {hN: tuple(list(hV) + [0.5]) for hN, hV in huePalette.items()}
        height, width = 3, 3
        aspect = width / height
        ################################
        # TODO: I don't remember why I recalculate the relative stats table here
        plotRelativeStatsDF = relativeStatsDF.reset_index()
        plotRelativeStatsDF.loc[:, 'electrode'] = plotRelativeStatsDF['electrode'].astype(spinalElecCategoricalDtype)
        plotRelativeStatsDF = plotRelativeStatsDF.loc[plotRelativeStatsDF['stimCondition'] != 'NA_0.0', :]
        plotRelativeStatsDF.loc[:, 'sigAnn'] = plotRelativeStatsDF.apply(
            lambda x: '{}\n '.format(x['feature'].replace('#0', '')) + r"$\bf{(*)}$" if x['reject'] else '{}\n '.format(
                x['feature'].replace('#0', '')), axis='columns')
        if plotTDist:
            g = sns.displot(
                x='T', hue='freqBandName',
                data=plotRelativeStatsDF,
                col=colVar,
                col_order=[
                    cn
                    for cn in colOrder
                    if cn in plotRelativeStatsDF[colVar].to_list()],
                row=hueName,
                row_order=[
                    cn
                    for cn in hueOrder
                    if cn in plotRelativeStatsDF[hueName].to_list()],
                kind='hist', element='step', stat='density',
                height=2 * height, aspect=aspect,
                facet_kws=dict(sharey=False)
                )
            for anName, ax in g.axes_dict.items():
                fillMin, fillMax = (
                    plotRelativeStatsDF['critical_T_min'].mean(),
                    plotRelativeStatsDF['critical_T_max'].mean()
                    )
                ax.axvspan(fillMin, fillMax, color='r', alpha=0.1, zorder=-100)
            g.suptitle('T-statistic distribution')
            g.tight_layout(pad=styleOpts['tight_layout.pad'])
            if not consoleDebug:
                pdf.savefig(bbox_inches='tight', pad_inches=0,)
            if arguments['showFigures']:
                plt.show()
            else:
                plt.close()
            #############
        if arguments['plotTopoEffectSize']:
            statPalettes = [
                sns.diverging_palette(220, 20, as_cmap=True),
                sns.diverging_palette(145, 300, s=60, as_cmap=True)
                ]
            for name, statsThisFB in plotRelativeStatsDF.groupby(['freqBandName', 'isMahalDist']):
                freqBandName, isMahalDist = name
                print('plotting topo {}'.format(name))
                try:
                    statsThisFB = statsThisFB.loc[(statsThisFB['stimCondition'] != 'NA_0.0').to_numpy(), :]
                    numStimC = statsThisFB['stimCondition'].unique().size
                    numKinC = statsThisFB['kinematicCondition'].unique().size
                    for statIdx, statName in enumerate(['hedges', 'T']):
                        fig, ax = plt.subplots(
                            numKinC, numStimC + 1,
                            figsize = (6 * numKinC, 6 * numStimC + .6),
                            gridspec_kw={
                                'width_ratios': [10] * numStimC + [1],
                                'wspace': 0.1}
                            )
                        vMin, vMax = statsThisFB[statName].min(), statsThisFB[statName].max()
                        cBarKinIdx = int(numKinC / 2)
                        cBarStimIdx = int(numStimC)
                        for kinIdx, stimIdx in product(range(numKinC), range(numStimC)):
                            kinName = np.unique(statsThisFB['kinematicCondition'])[kinIdx]
                            stimName = np.unique(statsThisFB['stimCondition'])[stimIdx]
                            thisMask = (statsThisFB['kinematicCondition'] == kinName) & (statsThisFB['stimCondition'] == stimName)
                            ann2D = statsThisFB.loc[thisMask, :].pivot(index='yCoords', columns='xCoords', values='sigAnn')
                            stats2D = statsThisFB.loc[thisMask, :].pivot(index='yCoords', columns='xCoords', values=statName)
                            heatMapKWs = dict(
                                vmin=vMin, vmax=vMax, center=0.,  fmt='s',
                                linewidths=0, cmap=statPalettes[statIdx],
                                annot=ann2D,
                                annot_kws=dict(fontsize=sns.plotting_context()["font.size"]),
                                xticklabels=False, yticklabels=False, square=True
                                )
                            if (kinIdx == cBarKinIdx) and (stimIdx == (cBarStimIdx - 1)):
                                heatMapKWs['cbar'] = True
                                heatMapKWs['cbar_ax'] = ax[cBarKinIdx, cBarStimIdx]
                            else:
                                heatMapKWs['cbar'] = False
                            sns.heatmap(
                                data=stats2D, ax=ax[kinIdx, stimIdx], ** heatMapKWs)
                            ax[kinIdx, stimIdx].set_title('{}, {}'.format(kinName, stimName))
                            ax[kinIdx, stimIdx].set_xlabel('')
                            ax[kinIdx, stimIdx].set_ylabel('')
                        for kinIdx in range(numKinC):
                            if kinIdx != cBarKinIdx:
                                ax[kinIdx, cBarStimIdx].set_xticks([])
                                ax[kinIdx, cBarStimIdx].set_yticks([])
                                sns.despine(
                                    fig=fig, ax=ax[kinIdx, cBarStimIdx],
                                    top=True, right=True, left=True, bottom=True,
                                    offset=None, trim=False)
                            else:
                                ax[kinIdx, cBarStimIdx].set_ylabel('{}'.format(statName))
                        figTitle = fig.suptitle('{} ({})'.format(statName, freqBandName))
                        fig.tight_layout(pad=styleOpts['tight_layout.pad'])
                        if not consoleDebug:
                            pdf.savefig(bbox_inches='tight', pad_inches=0, bbox_extra_artists=[figTitle],)
                        if arguments['showFigures']:
                            plt.show()
                        else:
                            plt.close()
                except Exception:
                    traceback.print_exc()
print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
