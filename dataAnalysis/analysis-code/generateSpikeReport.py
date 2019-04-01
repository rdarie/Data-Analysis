from currentExperiment import *
import neo
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
from importlib import reload

dataBlock = preproc.loadWithArrayAnn(spikePath, fromRaw=False)
allSpikeTrains = [
        i for i in dataBlock.filter(objects=SpikeTrain) if '#' in i.name]
spikes = preproc.spikeTrainsToSpikeDict(allSpikeTrains)
spikes['Units'] = 'a.u. (z-score)'
reportName = 'tdc_' + trialFilesFrom['utah']['ns5FileName'] + '_spike_report'
ssplt.spikePDFReport(
    figureFolder,
    spikes, cmpDF,
    arrayName='utah', arrayInfo=trialFilesFrom['utah'],
    rasterOpts=rasterOpts, plotOpts=plotOpts,
    trialStats=None, newName=reportName)
