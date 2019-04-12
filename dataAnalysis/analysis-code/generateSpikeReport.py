#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    generateSpikeReport [--trialIdx=trialIdx]

Options:
    --trialIdx=trialIdx            which trial to analyze
"""

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
import os
from docopt import docopt
arguments = docopt(__doc__)
#  if overriding currentExperiment
if arguments['--trialIdx']:
    print(arguments)
    trialIdx = int(arguments['--trialIdx'])
    ns5FileName = 'Trial00{}'.format(trialIdx)
    spikePath = os.path.join(
        nspFolder,
        'tdc_' + ns5FileName,
        'tdc_' + ns5FileName + '.nix')
    trialFilesFrom = {
        'utah': {
            'origin': 'mat',
            'experimentName': experimentName,
            'folderPath': nspFolder,
            'ns5FileName': ns5FileName,
            'elecIDs': list(range(1, 97)) + [135],
            'excludeClus': []
            }
        }

dataBlock = preproc.loadWithArrayAnn(spikePath, fromRaw=False)
allSpikeTrains = [
        i for i in dataBlock.filter(objects=SpikeTrain) if '#' in i.name]
spikes = preproc.spikeTrainsToSpikeDict(allSpikeTrains)
spikes['Units'] = 'a.u. (z-score)'
reportName = 'tdc_' + ns5FileName + '_spike_report'
ssplt.spikePDFReport(
    spikeSortingFiguresFolder,
    spikes, cmpDF,
    arrayName='utah', arrayInfo=trialFilesFrom['utah'],
    rasterOpts=rasterOpts, plotOpts=plotOpts,
    trialStats=None, newName=reportName)
