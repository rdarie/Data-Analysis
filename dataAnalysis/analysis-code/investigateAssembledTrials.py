import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy) 
import dataAnalysis.helperFunctions.kilosort_analysis as ksa
import dataAnalysis.helperFunctions.helper_functions as hf
import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as preproc
import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from currentExperiment import *
import quantities as pq

dataReader = neo.io.nixio_fr.NixIO(
    filename=experimentDataPath)
binnedReader = neo.io.nixio_fr.NixIO(
    filename=experimentBinnedSpikePath)

masterSpikeMats, _ = preproc.loadSpikeMats(
    experimentBinnedSpikePath, rasterOpts,
    chans=['elec44#0'],
    loadAll=True)

segIdx = 3
byIndex = reader.get_analogsignal_chunk(
    block_index=0, seg_index=segIdx,
    i_start=1000000, i_stop=1100000,
    channel_ids=chanIdx)
byName = reader.get_analogsignal_chunk(
    block_index=0, seg_index=segIdx,
    i_start=1000000, i_stop=1100000,
    channel_names=chanNames)
plt.plot(byName, '--', label='byName', lw=5)
plt.plot(byIndex, label='byIndex')
plt.legend()
plt.show()
