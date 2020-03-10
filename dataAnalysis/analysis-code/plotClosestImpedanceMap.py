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
    --selectorName=selectorName            filename for resulting selector [default: minfrmaxcorr]
"""
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('Agg')   # generate postscript output
matplotlib.use('QT5Agg')   # generate postscript output
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook", font_scale=1.2)
sns.set_style("white")
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
import dataAnalysis.preproc.ns5 as ns5
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

analysisSubFolder = os.path.join(
    scratchFolder, arguments['analysisName']
    )
if not os.path.exists(analysisSubFolder):
    os.makedirs(analysisSubFolder, exist_ok=True)
#
if arguments['processAll']:
    prefix = assembledName
else:
    prefix = ns5FileName
resultPath = os.path.join(
    alignSubFolder,
    prefix + '_{}_{}_calc.h5'.format(
        arguments['inputBlockName'], arguments['window']))
selectorPath = os.path.join(
    analysisSubFolder,
    prefix + '_{}.pickle'.format(
        arguments['selectorName']))
#
def findImpedanceIfPresent(label):
    if impedances.index.contains(label):
        return impedances.loc[label, 'impedance']
    else:
        return np.nan
cmpDF['impedance'] = cmpDF['label'].map(findImpedanceIfPresent)
cmpDF['annotation'] = cmpDF.apply(lambda r: '{}\n {:.0f}k'.format(r['label'], r['impedance']), axis=1)
cmpDF.dropna(inplace=True, subset=['impedance'])
impedanceLong = cmpDF.pivot('xcoords', 'ycoords', 'impedance')
annotationLong = cmpDF.pivot('xcoords', 'ycoords', 'annotation')

f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(
    impedanceLong, annot=annotationLong, fmt='s',
    annot_kws={"size": 5}, linewidths=.5, ax=ax)
plt.savefig(os.path.join(figureFolder, 'impedanceMap.pdf'))
plt.close()
# plt.show()