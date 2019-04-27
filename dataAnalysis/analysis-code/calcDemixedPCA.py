import dataAnalysis.plotting.aligned_signal_plots as asp
import os, pdb, traceback
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as preproc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
#  load options
from currentExperiment import *
from joblib import dump, load
import quantities as pq

sns.set_style("whitegrid")
sns.set_context("poster")


def getConditionAverages(dataBlock, unitName, dataQueryTemplate):
    thisUnit = dataBlock.filter(
        objects=Unit, name=unitName)[0]
    waveformsDict = {}
    uniqueSpiketrains = []
    for i, st in enumerate(thisUnit.spiketrains):
        if st not in uniqueSpiketrains:
            uniqueSpiketrains.append(st)
            waveformsDict.update({
                i: preproc.spikeTrainWaveformsToDF([st])})
    allWaveforms = pd.concat(
        waveformsDict,
        names=['segment'] + waveformsDict[0].index.names)
    allWaveforms.columns.name = 'bin'
    allWaveformsWide = allWaveforms.stack().reset_index(name='signal')
    dataQuery = dataQueryTemplate.format(unitName)
    aveDF = allWaveformsWide.query(dataQuery)
    aveDF.loc[aveDF['pedalSizeCat'] == 'XL', 'pedalSizeCat'] = 'L'
    aveDF.loc[aveDF['pedalSizeCat'] == 'XS', 'pedalSizeCat'] = 'S'
    for idx, (name, group) in enumerate(aveDF.groupby(['segment', 'index'])):
        if group['amplitudeFuzzy'].sum() == 0:
            aveDF.loc[group.index, 'programFuzzy'] = idx % 3
    aveDF.set_index(aveDF.columns.drop(['bin', 'signal']).tolist(), inplace=True)
    returnDF = aveDF.pivot(columns='bin')['signal']
    return returnDF, dataQuery


dataBlock = preproc.loadWithArrayAnn(
    experimentTriggeredPath)

frUnits = [
    i
    for i in dataBlock.filter(objects=Unit)
    if '_fr' in i.name]

#  response to movement
dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalSizeCat==\'M\')',
    '(amplitudeCat==0)',
    '(bin>2750)', '(bin<7250)',
    ])

for thisUnit in frUnits:
    unitName = thisUnit.name
    aveDF, dataQuery = getConditionAverages(dataBlock, unitName, dataQueryTemplate)
    break

#  demixed PCA
dimensionsList = [
    'RateInHzFuzzy', 'amplitudeCatFuzzy',
    'pedalSizeCat', 'programFuzzy'
    ]

dataQueryTemplate = '&'.join([
    '(feature==\'{}\')',
    '(pedalMovementCat==\'midPeak\')',
    '(pedalSizeCat==\'M\')',
    '(bin>2750)', '(bin<7250)',
    ])

for thisUnit in frUnits:
    unitName = thisUnit.name
    aveDF, dataQuery = getConditionAverages(dataBlock, unitName, dataQueryTemplate)
    nObs = 0
    for name, group in aveDF.groupby(dimensionsList):
        nObs += group.shape[0]
        print('{}, {} observation(s)'.format(name, group.shape[0]))
    print('total: {}'.format(nObs))
    break

# diagnostics
plotDF = aveDF.stack().reset_index(name='signal').query('bin==5000')
ax = sns.distplot(plotDF['pedalMovementDuration'])

plt.savefig(
    os.path.join(
        figureFolder, 'debug.pdf'))
plt.close()