"""  13: Plot Firing Rates and Rasters aligned to Stim
Usage:
    temp.py [options]

Options:
    --trialIdx=trialIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --verbose                       print diagnostics? [default: True]
    --alignQuery=alignQuery         what will the plot be aligned to? [default: (pedalMovementCat==\'midPeak\')]
    --window=window                 process with short window? [default: short]
    --unitQuery=unitQuery           how to restrict channels? [default: (chanName.str.contains(\'pca\'))]
    --blockName=blockName           which trig_ block to pull [default: pca]
    --selector=selector             filename if using a unit selector
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib import pyplot as plt
import seaborn as sns
import os, pdb
from importlib import reload
import neo
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from copy import copy
# import dataAnalysis.helperFunctions.kilosort_analysis_new as ksa
import dataAnalysis.helperFunctions.helper_functions_new as hf
# import rcsanalysis.packet_func as rcsa_helpers
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.profiling as prf
# import dataAnalysis.preproc.mdt as preprocINS
import numpy as np
import pandas as pd
from collections import Iterable
import dill as pickle
#  load options
# import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib.lines import Line2D
# import matplotlib.animation as animation
# from matplotlib.animation import FFMpegWriter
# from collections import OrderedDict

from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['trialIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

#plt.style.use("dark_background")

sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")

if arguments['processAll']:
    prefix = experimentName
else:
    prefix = ns5FileName
triggeredPath = os.path.join(
    scratchFolder,
    prefix + '_{}_{}.nix'.format(
        arguments['blockName'], arguments['window']))
print('loading {}'.format(triggeredPath))
dataBlock = ns5.loadWithArrayAnn(triggeredPath)
otherBlock = ns5.loadWithArrayAnn(triggeredPath.replace(arguments['blockName'], 'other'))
# during movement and stim
pedalSizeQuery = '(' + '|'.join([
    '(pedalSizeCat == \'{}\')'.format(i)
    for i in ['M', 'L', 'XL']
    ]) + ')'

dataQuery = '&'.join([
    '((RateInHzFuzzy==100)|(RateInHzFuzzy==0))',
    #  '((amplitudeCatFuzzy>=2)|(amplitudeCatFuzzy==0))',
    pedalSizeQuery,
    #  '(pedalDirection == \'CW\')'
    ])

if arguments['alignQuery'] is not None:
    if len(arguments['alignQuery']):
        dataQuery = '&'.join([
            dataQuery,
            arguments['alignQuery'],
            ])
if not len(arguments['unitQuery']):
    arguments['unitQuery'] = None
colorPal = "ch:0.6,-.2,dark=.2,light=0.7,reverse=1"  #  for firing rates

if arguments['selector'] is not None:
    with open(
        os.path.join(
            scratchFolder,
            arguments['selector'] + '.pickle'),
            'rb') as f:
        selectorMetadata = pickle.load(f)
    chanNames = [
        i.replace('_raster#0', '')
        for i in selectorMetadata['outputFeatures']]
else:
    chanNames = None

alignedAsigsKWargs = dict(
    duplicateControlsByProgram=False,
    makeControlProgram=True,
    amplitudeColumn='amplitudeFuzzy',
    programColumn='programFuzzy',
    electrodeColumn='electrodeFuzzy',
    transposeToColumns='feature', concatOn='columns',
    removeFuzzyName=False)

pcList = ['pca{:0>1}#0'.format(i) for i in range(3)]
masterSpikeMat = ns5.alignedAsigsToDF(
    dataBlock, pcList,
    dataQuery=dataQuery, verbose=True,
    getMetaData=True, decimate=5,
    **alignedAsigsKWargs)
otherMat = ns5.alignedAsigsToDF(
    otherBlock, chanNames,
    dataQuery=dataQuery, verbose=True,
    getMetaData=True, decimate=5,
    **alignedAsigsKWargs)
prf.print_memory_usage('just loaded firing rates')

winWidth = 40
nFrames = 999
dataQuery = '&'.join(['((amplitudeCatFuzzy == 0)|(amplitudeCatFuzzy == 3))', '(bin>-2.5)', ('bin<2.5')])
unpackedFeatures = pd.concat([masterSpikeMat[pcList], otherMat[['amplitude#0', 'position#0']]], axis='columns')
sns.set_style("darkgrid", {'axes.facecolor': '0.7', 'figure.facecolor': '0.7'})
plotKws = {'linewidth': 2}
unpackedFeatures['position#0'] = unpackedFeatures['position#0'] * 100

fig = plt.figure(figsize=(3, 2), dpi=300)
#, tight_layout={'h_pad': 0, 'w_pad': 1.5}
featAx = plt.subplot2grid((1, 3), (0, 0), colspan=2, projection='3d', aspect='auto')
moveAx = plt.subplot2grid((1, 3), (0, 2), projection='3d', aspect='equal')
aniFeat = hf.animateDFSubset3D(
    unpackedFeatures, dataQuery, winWidth, nFrames,
    xyzList=pcList, pltKws=plotKws, ax=featAx,
    colorCol='amplitude#0')
aniPedal = hf.animateAngle3D(
    unpackedFeatures, dataQuery, winWidth, nFrames, ax=moveAx,
    pltKws=plotKws, saveToFile=None, extraAni=[aniFeat])
plt.show()
# os.path.join(figureFolder, 'move.mp4')