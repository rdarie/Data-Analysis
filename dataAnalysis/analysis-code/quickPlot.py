"""
Usage:
    temp.py [options]

Options:
    --exp=exp                              which experimental day to analyze
    --blockIdx=blockIdx                    which trial to analyze [default: 1]
    --processAll                           process entire experimental day? [default: False]
    --lazy                                 load from raw, or regular? [default: False]
    --verbose                              print diagnostics? [default: False]
    --window=window                        process with short window? [default: long]
    --inputBlockName=inputBlockName        filename for inputs [default: fr]
    --analysisName=analysisName            append a name to the resulting blocks? [default: default]
    --alignFolderName=alignFolderName      append a name to the resulting blocks? [default: motion]
    --selector=selector                    filename for resulting selector [default: minfrmaxcorr]
"""
import os
#  import numpy as np
#  import pandas as pd
import pdb
import re
from datetime import datetime as dt
import numpy as np
#  from neo import (
#      Block, Segment, ChannelIndex,
#      Event, AnalogSignal, SpikeTrain, Unit)
#  import neo
import dill as pickle
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import pandas as pd
from neo.io import NixIO, nixio_fr, BlackrockIO
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
from neo.core import (Block, Segment, ChannelIndex,
    AnalogSignal, Unit, SpikeTrain, Event)
import dataAnalysis.preproc.ns5 as ns5
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

rawBasePath = os.path.join(
    nspFolder, 'Block001.ns5')
reader = BlackrockIO(
    filename=rawBasePath, nsx_to_load=5)
dataBlock = ns5.readBlockFixNames(
    reader,
    block_index=0, signal_group_mode='split-all',
    lazy=False)

rawPairs = [
    [3, 4],
    [7, 6],
    [13, 14],
    [11, 12],
    [9, 10],
    [17, 18],
    [19, 22],
    [21, 24],
]
pairs = [
    [j - 1 for j in i]
    for i in rawPairs
] + [
    [j - 1 + 32 for j in i]
    for i in rawPairs
]
aSigs = dataBlock.filter(objects=AnalogSignal)
fs = 30000
w0 = 60
bw = 4
Wn = [
    2 * i / fs
    for i in [w0 - bw/2, w0 + bw/2]]
from scipy import signal
sos = signal.butter(
    4, Wn=Wn, btype='bandstop',
    analog=False, output='sos')

for sigPair in pairs:
    rerefAsig = (
        dataBlock.segments[0].analogsignals[sigPair[0]] -
        dataBlock.segments[0].analogsignals[sigPair[1]])
    rerefAsig.magnitude[:, 0] = signal.sosfiltfilt(
        sos, rerefAsig.magnitude[:, 0])
    dataBlock.segments[0].analogsignals[sigPair[0]] = rerefAsig

savePath = os.path.join(
    nspFolder, 'Block001.nix')
writer = NixIO(filename=savePath)
writer.write_block(dataBlock, use_obj_names=True)
writer.close()