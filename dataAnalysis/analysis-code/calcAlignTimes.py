import dataAnalysis.ephyviewer.scripts as vis_scripts
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from copy import copy
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import KFold
from sklearn.model_selection import fit_grid_point
from sklearn.metrics import mean_squared_error, r2_score
from collections import Iterable
#  load options
from currentExperiment import *
from joblib import dump, load

#  all experimental days
#  use for elec names as workaround
insDataPath = os.path.join(
    trialFilesStim['ins']['folderPath'],
    trialFilesStim['ins']['experimentName'],
    'Trial003_ins.nix')
dataBlock = preproc.loadWithArrayAnn(
    experimentDataPath, fromRaw=False)
insBlock = preproc.loadWithArrayAnn(
    insDataPath, fromRaw=False)
#  passing binnedspikepath interpolates down to 1 msec
tdDF, stimStatus = preproc.unpackAnalysisBlock(
    dataBlock, interpolateToTimeSeries=True,
    binnedSpikePath=binnedSpikePath)
#  todo: get from binnedSpikes
fs = (1 / (tdDF['t'].iloc[1] - tdDF['t'].iloc[0]))
#  some categories need to be calculated, others are available
fuzzyCateg = ['amplitude', 'program', 'RateInHz']
availableCateg = ['movement', 'moveCat', 'ampCat']
calcFromTD = ['stimOffset']
categoryColumns = calcFromTD + availableCateg + fuzzyCateg

#  calculate movement peak:
atRest = (tdDF['movement'] == 1) & (tdDF['movement'].shift(-1) == 0)
restPosition = tdDF.loc[atRest, 'position'].mean()
atPeak = (
    ((tdDF['position'] - restPosition).abs() > 0.01) &
    (tdDF['movement'] == 0))
tdDF['movePeak'] = np.nan
tdDF.loc[atPeak, 'movePeak'] = tdDF.loc[atPeak, 'position']
tdDF['movePeak'].interpolate(method='nearest', inplace=True)
#  asign movement peak category
#  ax = sns.distplot(tdDF['movePeak'].fillna(0))
#  plt.show()
catBins = [-0.9, -0.45, -0.2, 0.2, 0.55, 0.9]
tdDF['ampCat'] = pd.cut(
    tdDF['movePeak'], catBins,
    labels=['XL', 'L', 'M', 'S', 'XS'])

#  get alignment times
movingAtAll = tdDF['movement'].fillna(0).abs()
movementOnOff = movingAtAll.diff()
moveMask = movementOnOff == 1
moveTimes = tdDF.loc[
    moveMask, 't']

stopMask = movementOnOff == -1
stopTimes = tdDF.loc[
    stopMask, 't']
# if stopped before the first start, drop it
dropIndices = stopTimes.index[stopTimes < moveTimes.iloc[0]]
stopTimes.drop(index=dropIndices, inplace=True)

tdDF['moveCat'] = np.nan
tdDF.loc[
    moveMask & (tdDF['movement'] == 1),
    'moveCat'] = 'return'
tdDF.loc[
    moveMask & (tdDF['movement'] == -1),
    'moveCat'] = 'outbound'
tdDF.loc[
    stopMask & (tdDF['movement'].shift(1) == -1),
    'moveCat'] = 'reachedPeak'
tdDF.loc[
    stopMask & (tdDF['movement'].shift(1) == 1),
    'moveCat'] = 'reachedBase'
#  tdDF['moveCat'].value_counts()
#  get intervals halfway between move stop and move start
pauseLens = moveTimes.shift(-1).values - stopTimes
maskForLen = pauseLens > 1.5

halfOffsets = (fs * (pauseLens / 2)).fillna(0).astype(int)

otherTimesIdx = (stopTimes.index + halfOffsets.values)[maskForLen]
otherTimes = tdDF.loc[otherTimesIdx, 't']

moveCategories = tdDF.loc[
    tdDF['t'].isin(moveTimes), fuzzyCateg + availableCateg
    ].reset_index(drop=True)
moveCategories['metaCat'] = 'onset'
stopCategories = tdDF.loc[
    tdDF['t'].isin(stopTimes), fuzzyCateg + availableCateg
    ].reset_index(drop=True)
stopCategories['metaCat'] = 'offset'
otherCategories = tdDF.loc[
    tdDF['t'].isin(otherTimes), fuzzyCateg + availableCateg
    ].reset_index(drop=True)
otherCategories['metaCat'] = 'rest'

otherCategories['program'] = 999
otherCategories['RateInHz'] = 999
otherCategories['ampCat'] = 'Control'
otherCategories['moveCat'] = 'Control'

alignTimes = pd.concat((
    moveTimes, stopTimes, otherTimes),
    axis=0, ignore_index=True)
#  sort align times
alignTimes.sort_values(inplace=True, kind='mergesort')
categories = pd.concat((
    moveCategories, stopCategories, otherCategories),
    axis=0, ignore_index=True)
#  sort categories by align times
#  (needed to propagate values forward)
categories = categories.loc[alignTimes.index, :]
alignTimes.reset_index(drop=True, inplace=True)
categories.reset_index(drop=True, inplace=True)
    
for colName in calcFromTD:
    categories[colName] = np.nan

#  pdb.set_trace()
confirmationPlot = False
progAmpNames = rcsa_helpers.progAmpNames
plotCols = ['movePeak', 'position'] + progAmpNames
eventCols = ['metaCat', 'ampCat', 'moveCat', 'amplitude']
if confirmationPlot:
    confStart = 300
    confStop = 400
    tdMask = (tdDF['t'] > confStart) & (tdDF['t'] < confStop)
    for pC in plotCols:
        plotTrace = tdDF.loc[tdMask, pC]
        if pC == 'position':
            plotTrace = plotTrace * (-1) 
        if plotTrace.max() > 1:
            plotTrace = plotTrace / plotTrace.max()
        plt.plot(tdDF.loc[tdMask, 't'], plotTrace, label=pC)
    alignMask = (alignTimes > confStart) & (alignTimes < confStop)
    maskedCat = categories.loc[alignMask, :]
    maskedTimes = alignTimes.loc[alignMask]
    for i, eC in enumerate(eventCols):
        for j, catOption in enumerate(pd.unique(maskedCat[eC])):
            catMask = maskedCat[eC] == catOption
            plt.plot(
                maskedTimes[catMask],
                maskedTimes[catMask] * 0,
                'o', label=catOption, alpha=0.5)
            for x in maskedTimes[catMask]:
                plt.text(x, (10*i + j) / 30, '{}'.format(catOption))
    #  plt.plot(tdDF.loc[tdMask, 't'], tdDF.loc[tdMask, 'movement'], 'o', label='movement')
    plt.legend()
    plt.show()


print('Before fuzzy correction: amplitude counts')
print(categories['amplitude'].value_counts())
print('program counts')
print(categories['program'].value_counts())
# fudgeFactor to account for stim and move not lining up
fudgeFactor = 300e-3  # seconds
for idx, tOnset in alignTimes.iteritems():
    moveCat = categories.loc[idx, 'moveCat']
    metaCat = categories.loc[idx, 'metaCat']
    if metaCat == 'onset':
        tStart = max(0, tOnset - fudgeFactor)
        tStop = min(tdDF['t'].iloc[-1], tOnset + fudgeFactor)

        tdMaskPre = (tdDF['t'] > tStart) & (tdDF['t'] < tOnset)
        tdMaskPost = (tdDF['t'] > tOnset) & (tdDF['t'] < tStop)
        tdMask = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)
        
        theseAmps = tdDF.loc[tdMask, ['t', 'amplitude']]
        ampDiff = theseAmps.diff()

        ampOnset = theseAmps.loc[ampDiff[ampDiff['amplitude'] > 0].index, 't']
        if len(ampOnset):
            # greater if movement after stim
            categories.loc[idx, 'stimOffset'] = tOnset - ampOnset.iloc[0]

        ampOffset = theseAmps.loc[ampDiff[ampDiff['amplitude'] < 0].index, 't']
        
        #  if there's an amp offset, use the last value where amp was on
        if len(ampOffset):
            fuzzyIdx = ampDiff[ampDiff['amplitude'] < 0].index[0] - 1
        else:
            fuzzyIdx = tdDF.loc[tdMask, :].index[-1]
        #  pdb.set_trace()
        for colName in fuzzyCateg:
            nominalValue = categories.loc[idx, colName]
            fuzzyValue = tdDF.loc[fuzzyIdx, colName]

            if (nominalValue != fuzzyValue):
                categories.loc[idx, colName] = fuzzyValue
                print('nominally, {} is {}'.format(colName, nominalValue))
                print('changed it to {}'.format(fuzzyValue))
    elif metaCat == 'offset':
        #  offsets inherit the amplitude of the onset
        for colName in fuzzyCateg:
            categories.loc[idx, colName] = np.nan


categories.fillna(method='ffill', inplace=True)
#  rename all controls
amp0Mask = (categories['amplitude'] == 0) & (categories['program'] == 0)
categories.loc[amp0Mask, 'program'] = 999
categories.loc[amp0Mask, 'RateInHz'] = 999

print('after fuzzy correction: amplitude counts')
print(categories['amplitude'].value_counts())
print('program counts')
print(categories['program'].value_counts())

#  plotting stuff
confirmationPlot = False
if confirmationPlot:
    for idx, tOnset in alignTimes.iteritems():
        tStart = max(0, tOnset - fudgeFactor)
        tStop = min(tdDF['t'].iloc[-1], tOnset + fudgeFactor)

        tdMaskPre = (tdDF['t'] > tStart) & (tdDF['t'] < tOnset)
        tdMaskPost = (tdDF['t'] > tOnset) & (tdDF['t'] < tStop)
        tdMask = (tdDF['t'] > tStart) & (tdDF['t'] < tStop)
        moveCat = categories.loc[idx, 'moveCat']
        metaCat = categories.loc[idx, 'metaCat']    
        if metaCat == 'offset':
            #  plotting toggle
            progAmpNames = rcsa_helpers.progAmpNames
            plotCols = ['program', 'position'] + progAmpNames
            for pC in plotCols:
                plotTrace = tdDF.loc[tdMask, pC]
                if plotTrace.max() > 1:
                    plotTrace = plotTrace / plotTrace.max()
                plt.plot(tdDF.loc[tdMask, 't'] - tOnset, plotTrace, label=pC)
            #  plt.plot( 
            #      tdDF.loc[fuzzyIdx, 't'] - tOnset,
            #      fuzzyValue,
            #      'o', label='fuzzyValue')
            titleStr = ''
            titleCategories = [
                'program', 'amplitude',
                'moveCat', 'ampCat', 'metaCat']
            for name, entry in categories.loc[idx, titleCategories].items():
                titleStr += '{} '.format(name)
                titleStr += '{} '.format(entry)
            plt.title(titleStr)
            plt.legend()
            plt.show()
            #  plt.show(block=False)
            #  plt.pause(3)
            #  plt.close()

#  fix program labels
#  INS amplitudes are in 100s of uA
categories['amplitude'] = categories['amplitude'] / 10
#  pull actual electrode names
categories['electrode'] = np.nan

for pName in pd.unique(categories['program']):
    pMask = categories['program'] == pName
    if pName == 999:
        categories.loc[pMask, 'electrode'] = 'Control'
    else:
        unitName = 'g0p{}'.format(int(pName))
        thisUnit = insBlock.filter(objects=Unit, name=unitName)[0]
        cathodes = thisUnit.annotations['cathodes']
        anodes = thisUnit.annotations['anodes']
        elecName = ''
        if isinstance(anodes, Iterable):
            elecName += '+ ' + ', '.join(['E{}'.format(i) for i in anodes])
        else:
            elecName += '+ E{}'.format(anodes)
        elecName += ' '
        if isinstance(cathodes, Iterable):
            elecName += '- ' + ', '.join(['E{}'.format(i) for i in cathodes])
        else:
            elecName += '- E{}'.format(cathodes)
        categories.loc[pMask, 'electrode'] = elecName

print('Electrode Count')
print(categories['electrode'].value_counts())

offsetPlot = False
if offsetPlot:
    ax = sns.distplot(categories['stimOffset'].dropna())
    plt.show()
chooseTDChans = ['ins_td0', 'position', 'amplitude']
spikeMats, validTrials = preproc.loadSpikeMats(
    binnedSpikePath, rasterOpts, alignTimes)
posMats, _ = preproc.loadSpikeMats(
    experimentDataPath, rasterOpts, alignTimes, chans=chooseTDChans)
#  plt.imshow(rawSpikeMat.values, aspect='equal'); plt.show()
checkOldIsNew = False
if checkOldIsNew:
    allSpikeTrains = [
        i for i in dataBlock.filter(objects=SpikeTrain) if 'ch' in i.name]
    spikes = preproc.spikeTrainsToSpikeDict(allSpikeTrains)
    spikeMatsOld = hf.binnedArray(spikes, rasterOpts, alignTimes, chans=None)
    matIdx = 7
    chanName = 'ch20#0'
    fig, ax = plt.subplots(1, 2)
    spikeMats[matIdx].loc[chanName, :].plot(ax=ax[0])
    ax[0].set_title('New')
    chanIdx = spikes['ChannelID'].index(chanName)
    unitIdx = pd.unique(spikes['Classification'][chanIdx])[0]
    spikeMatsOld[matIdx].loc[unitIdx, :].plot(ax=ax[1])
    ax[1].set_title('Old')
    plt.show()
    fig, ax = plt.subplots(1, 2)
    newMat, spikeOrder = ksa.sortBinnedArray(spikeMats[matIdx], 'meanFR')
    hf.plotBinnedSpikes(newMat, show=False, ax=ax[0])
    ax[0].set_title('New')
    oldMat, spikeOrder = ksa.sortBinnedArray(spikeMatsOld[matIdx], 'meanFR')
    hf.plotBinnedSpikes(oldMat, show=False, ax=ax[1])
    ax[1].set_title('Old')
    plt.show()

whichTrials = alignTimes.index[validTrials]
spikeMat2D = pd.concat(spikeMats, names=['trial'])
posMat2D = pd.concat(posMats, names=['trial'])
posMat2D.rename(
    columns={
        'ins_td0': 'ins_td0',
        'position': 'position',
        'amplitude': 'tdAmplitude'
    }, inplace=True
)
#  INS amplitudes are in 100s of uA
posMat2D['tdAmplitude'] = posMat2D['tdAmplitude'] / 10

#  plt.imshow(masterSpikeMat2D.values, aspect='auto'); plt.show()
trainOnDifferentTimes = True
if trainOnDifferentTimes:
    estimator = load(estimatorPath)
    features = estimator.transform(spikeMat2D.values)
    nComp = 5
    compNames = ['PC{}'.format(i+1) for i in range(nComp)]
else:
    nComp = 5
    compNames = ['PC{}'.format(i+1) for i in range(nComp)]
    pca = PCA(n_components=nComp)
    estimator = Pipeline([('dimred', pca)])
    features = estimator.fit_transform(spikeMat2D.values)
    dump(estimator, estimatorPath)

featuresDF = pd.DataFrame(
    features, index=spikeMat2D.index, columns=compNames)
featuresDF = pd.concat((
    featuresDF, posMat2D
    ), axis=1)

#  featuresDF.columns.name = 'features'
#  index by trial to add categories

unpackedFeatures = featuresDF.reset_index()
for categName in categories.columns:
    unpackedFeatures[categName] = np.nan
    for trialIdx, group in unpackedFeatures.groupby('trial'):
        unpackedFeatures.loc[
            group.index, categName] = categories.loc[trialIdx, categName]

unpackedFeatures.to_hdf(featurePath, 'features')
