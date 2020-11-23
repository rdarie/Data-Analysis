#!/gpfs/runtime/opt/python/3.5.2/bin/python3
"""
Usage:
    generateSpikeReport [options]

Options:
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --exp=exp                                   which experimental day to analyze
    --nameSuffix=nameSuffix                     add anything to the output name?
    --arrayName=arrayName                       which electrode array to analyze [default: utah]
    --sourceFileSuffix=sourceFileSuffix         which source file to analyze
"""
import matplotlib
matplotlib.use('PS')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import neo
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.plotting.spike_sorting_plots as ssplt
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
from importlib import reload
import os, pdb

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

arrayName = arguments['arrayName']
electrodeMapPath = spikeSortingOpts[arrayName]['electrodeMapPath']
mapExt = electrodeMapPath.split('.')[-1]

if mapExt == 'cmp':
    cmpDF = prb_meta.cmpToDF(electrodeMapPath)
elif mapExt == 'map':
    cmpDF = prb_meta.mapToDF(electrodeMapPath)

if 'rawBlockName' in spikeSortingOpts[arrayName]:
    ns5FileName = ns5FileName.replace(
        'Block', spikeSortingOpts[arrayName]['rawBlockName'])
    triFolder = os.path.join(
        scratchFolder, 'tdc_{}{:0>3}'.format(
            spikeSortingOpts[arrayName]['rawBlockName'], blockIdx))
else:
    triFolder = os.path.join(
        scratchFolder, 'tdc_Block{:0>3}'.format(blockIdx))
if arguments['sourceFileSuffix'] is not None:
    triFolder = triFolder + '_{}'.format(arguments['sourceFileSuffix'])
    ns5FileName = ns5FileName + '_{}'.format(arguments['sourceFileSuffix'])

if not os.path.exists(spikeSortingFiguresFolder):
    os.makedirs(spikeSortingFiguresFolder, exist_ok=True)

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
    if (
        (len(i.units) > 0) &
        ('elec' in i.name) |
        ('utah' in i.name) |
        ('nform' in i.name)
        )]
#  [i.name for i in spChanIdx]
#  [len(i.units) for i in dataBlock.filter(objects=ChannelIndex)]
#  ['elec' in i.name for i in dataBlock.filter(objects=ChannelIndex)]
#  #)
spikes = preproc.channelIndexesToSpikeDict(spChanIdx)

spikes['Units'] = 'a.u. (z-score)'
reportName = 'tdc_' + ns5FileName + '_spike_report' + nameSuffix
spikeStruct = cmpDF.loc[cmpDF['elecName'] != 'ainp', :]
# spikeStruct.loc[:, 'label'] = [
#     i.replace('_', '.') + ' raw'
#     for i in spikeStruct['label']
#     ]
ssplt.spikePDFReport(
    spikeSortingFiguresFolder,
    spikes, spikeStruct,
    arrayName='utah', arrayInfo=trialFilesFrom['utah'],
    rasterOpts=rasterOpts, plotOpts=plotOpts,
    trialStats=None, newReportName=reportName, colorByAmpBank=True)
