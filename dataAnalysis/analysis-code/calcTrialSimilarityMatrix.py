"""
Usage:
    calcSimilarityMatrix.py [options]

Options:
    --trialIdx=trialIdx               which trial to analyze [default: 1]
    --exp=exp                         which experimental day to analyze [default: exp201901271000]
    --processAll                      process entire experimental day? [default: False]
    --lazy                            load from raw, or regular? [default: False]
    --window=window                   process with short window? [default: long]
    --analysisName=analysisName       append a name to the resulting blocks? [default: default]
    --unitQuery=unitQuery             how to select channels if not supplying a list? [default: neural]
    --alignQuery=alignQuery           query what the units will be aligned to? [default: midPeak]
    --selector=selector               filename if using a unit selector
    --eventName=eventName             name of events object to align to [default: motionStimAlignTimes]
"""
#  The text block above is used by the docopt package to parse command line arguments
#  e.g. you can call <python3 calcTrialSimilarityMatrix.py> to run with default arguments
#  but you can also specify, for instance <python3 calcTrialSimilarityMatrix.py --trialIdx=2>
#  to load trial 002 instead of trial001 which is the default
#
#  regular package imports
import os, pdb, traceback
from importlib import reload
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import numpy as np
import pandas as pd
import quantities as pq
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
#   you can specify options related to the analysis via the command line arguments,
#   or by saving variables in the currentExperiment.py file, or the individual exp2019xxxxxxxx.py files
#
#   these lines process the command line arguments
#   they produces a python dictionary called arguments
from namedQueries import namedQueries
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
#  this stuff imports variables from the currentExperiment.py and exp2019xxxxxxxx.py files
from currentExperiment import parseAnalysisOptions
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
#   once these lines run, your workspace should contain a bunch of variables that were imported
#   you can check them by calling the python functions globals() for global variables and locals() for local variables
#   both of these functions return python dictionaries
#
#   these lines change some options from the arguments, in case they were overriden by something written in the
#   currentExperiment.py and exp2019xxxxxxxx.py files
if (overrideChanNames is not None):
    arguments['unitNames'] = overrideChanNames
    arguments['unitQuery'] = None
else:
    arguments['unitNames'], arguments['unitQuery'] = ash.processUnitQueryArgs(
        namedQueries, scratchFolder, **arguments)
#
analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
verbose = True
#   figure out the path to the data
if arguments['processAll']:
    experimentDataPath = experimentDataPath.format(arguments['analysisName'])
    dataPath = experimentDataPath
    prefix = experimentName
else:
    analysisDataPath = analysisDataPath.format(arguments['analysisName'])
    dataPath = analysisDataPath
    prefix = ns5FileName
#
#   Anyway, the point is that we imported a long list of variables that contain information
#   related to the analysis - it's overly convoluted, no need to understand every detail
#
#   one of the options variables, rasteropts, contains the size of the window around every trial
#   i.e. how many seconds of data to select around the triggering time
#   since neo uses the quantities package (pq for short) to keep track of units of measurement,
#   here i am converting that window information into seconds
pdb.set_trace()
windowSize = [
    i * pq.s
    for i in rasterOpts['windowSizes'][arguments['window']]]
#
#   this function creates the data reader and reads in the block
#   arguments['lazy'] controls whether the neo block is fully loaded into memory
#   or loaded through the "lazy" method, you can read more about this in the neo documentation
dataReader, dataBlock = ns5.blockFromPath(
    dataPath, lazy=arguments['lazy'])
#   this statement peeks into the dataBlock and finds all the (neo.Unit)s that match the
#   "neural" query. There are other (neo.Units)s that, for instance, contain info about the
#   spinal cord stimulation pulse times; we don't want to analyze those as if they were
#   action potentials
if arguments['unitNames'] is None:
    arguments['unitNames'] = ns5.listChanNames(dataBlock, arguments['unitQuery'], objType=Unit)
#   you can inspect the .nix file using the NeuralEnsemble data viewer by calling
#   <python3 launchVis.py> from your analysis-code directory
#
#   the important data structures are the events (namely, the neo.Event named "motionStimAlignTimes")
#   which contains the timestamps in the experiment related to each trial (e.g. when did the pedal start moving)
#   and the metadata (how far did it move that trial, was there stimulation on, which electrodes were stimulating)
#   we need this information + the windowSize in order to determine when each trial started and ended
#
#   Then, the neo.Units that match the names in arguments['unitNames'] contain the action potential
#   timestamps for the entire recording. We need to scan through the event times and use them to select subsets
#   of the spiketrains. So,
#
#   1) Organize spike timestamps based on event timestamps
#       This is going to be the hard part!
#       I already have a function to do this for the AnalogSignals
#       you can look at it for reference (dataAnalysis.preproc.ns5.getAsigsAlignedToEvents())

#   2) calculate the similarity matrix (elephant implements this!)

#   3) t-sne the similarity matrix (scikit learn implements this!)

#   4) Discover new things about how spinal cord stimulation affects the brain, publish, become famous.