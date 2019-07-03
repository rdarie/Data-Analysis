#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    generateSpikeReport [options]

Options:
    --trialIdx=trialIdx            which trial to analyze [default: 1]
    --exp=exp                      which experimental day to analyze
    --nameSuffix=nameSuffix        add anything to the output name?
"""
import matplotlib
matplotlib.use('PS')
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
import os, pdb

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

if arguments['nameSuffix']:
    nameSuffix = arguments['nameSuffix']
else:
    nameSuffix = ''

spikePath = os.path.join(
    triFolder,
    'tdc_' + ns5FileName + '.nix')

dataBlock = preproc.loadWithArrayAnn(spikePath, fromRaw=False)

spChanIdx = [
    i
    for i in dataBlock.filter(objects=ChannelIndex)
    if ((len(i.units) > 0) & ('elec' in i.name))]
#  [i.name for i in spChanIdx]
#  [len(i.units) for i in dataBlock.filter(objects=ChannelIndex)]
#  ['elec' in i.name for i in dataBlock.filter(objects=ChannelIndex)]
#  pdb.set_trace()
spikes = preproc.channelIndexesToSpikeDict(spChanIdx)

spikes['Units'] = 'a.u. (z-score)'
reportName = 'tdc_' + ns5FileName + '_spike_report' + nameSuffix
spikeStruct = cmpDF[cmpDF['elecName'] == 'elec']

ssplt.spikePDFReport(
    spikeSortingFiguresFolder,
    spikes, spikeStruct,
    arrayName='utah', arrayInfo=trialFilesFrom['utah'],
    rasterOpts=rasterOpts, plotOpts=plotOpts,
    trialStats=None, newReportName=reportName)
