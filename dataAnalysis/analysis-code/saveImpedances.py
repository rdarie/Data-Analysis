"""
Usage:
    saveImpedances.py [options]

Options:
    --exp=exp                           which experimental day to analyze [default: exp201901271000]
    --processAll                        process all experimental days? [default: False]
    --verbose                           print diagnostics? [default: False]
    --plotting                          plot out the correlation matrix? [default: False]
    --ripple                            impedances saved with Trellis? [default: False]
    --blackrock                         impedances saved with Central? [default: False]
"""
#  The text block above is used by the docopt package to parse command line arguments
#  e.g. you can call <python3 calcBlockSimilarityMatrix.py> to run with default arguments
#  but you can also specify, for instance <python3 calcBlockSimilarityMatrix.py --blockIdx=2>
#  to load trial 002 instead of trial001 which is the default
#
#  regular package imports
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# matplotlib.use('PS')  # generate postscript output 
matplotlib.use('QT5Agg')  # generate postscript output
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("dark")
import matplotlib.ticker as ticker
import os, pdb, traceback
from importlib import reload
from neo import (
    Block, Segment, ChannelIndex,
    Event, AnalogSignal, SpikeTrain, Unit)
from neo.io.proxyobjects import (
    AnalogSignalProxy, SpikeTrainProxy, EventProxy)
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.aligned_signal_helpers as ash
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import numpy as np
import pandas as pd
import quantities as pq
import dill as pickle
from datetime import datetime as dt
import glob, re
import importlib
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
    1, arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

oldImpedances = pd.read_hdf('./impedances.h5', 'impedance')
impList = [oldImpedances]
if arguments['blackrock']:
    def stripImpedance(x):
        return int(x[:-5])

    def stripName(x):
        return re.split(r'\d*', x)[0]

    cmpDF = prb_meta.cmpToDF(nspCmpPath)
    utahLabels = cmpDF.loc[cmpDF['bank'].isin(['A', 'B', 'D']), 'label']
    if arguments['processAll']:
        fileList = glob.iglob('./**/impedances-blackrock.txt', recursive=True)
    else:
        fileList = []
    for filename in fileList:
        folderPath = filename.split('\\')[-2]
        print(folderPath)
        newImpedances = pd.read_csv(filename, sep='\t', skiprows=9, header=None, names=['elec', 'impedance'])
        newImpedances['impedance'] = newImpedances['impedance'].apply(stripImpedance)
        newImpedances['elec'] = newImpedances['elec'].apply(str.strip)
        newImpedances['elecType'] = np.nan
        newImpedances.loc[newImpedances['elec'].isin(utahLabels.values), 'elecType'] = 'utah'
        recDateStr = [i for i in (re.findall('(\d*)', filename)) if len(i)]
        if not len(recDateStr):
            continue
        recDate = dt.strptime(recDateStr[0], '%Y%m%d%H%M')
        newImpedances['date'] = recDate
        impList.append(newImpedances)
    allImpedances = pd.concat(impList)
    allImpedances.to_hdf('./impedances.h5', 'impedance')

if arguments['ripple']:
    headerNames = [
        'Array', 'Elec', 'Pin', 'Front End',
        'Freq(Hz)', 'Curr(nA)', 'Cycles', 'Mag(kOhms)', 'Phase']
    if arguments['processAll']:
        fileList = glob.iglob(
            os.path.join(
                remoteBasePath, 'raw', '*',
                'impedances-ripple.txt'),
            recursive=True)
    else:
        fileList = []
    for filename in fileList:
        recDateStr = [i for i in (re.findall('(\d*)', filename)) if len(i)]
        if not len(recDateStr):
            continue
        recDate = dt.strptime(recDateStr[0], '%Y%m%d%H%M')
        experimentShorthand = 'exp{}'.format(recDateStr[0])
        optsModule = importlib.import_module(experimentShorthand, package=None)
        expOpts = optsModule.getExpOpts()
        mapDF = prb_meta.mapToDF(expOpts['rippleMapFile'][int(arguments['blockIdx'])])
        folderPath = os.path.dirname(filename)
        newImpedances = pd.read_csv(
            filename, sep='\s+', skiprows=10,
            skipinitialspace=True,
            header=None, names=headerNames)
        
        def impedanceToNumeric(x):
            if x == '-':
                return np.nan
            else:
                return float(x)
        
        newImpedances['impedance'] = (
            newImpedances['Mag(kOhms)']
            .apply(impedanceToNumeric))
        pinBreakout = newImpedances['Pin'].str.split(':')
        feBreakout = pinBreakout.apply(lambda x: x[1]).str.split('-')
        processor = (
            pinBreakout
            .apply(lambda x: int(x[0][-1])))
        bank = (
            feBreakout
            .apply(lambda x: '{}.{}'.format(x[0], x[1])))
        FE = pd.Series('', index=newImpedances.index)
        for rowIdx, row in newImpedances.iterrows():
            FE.loc[rowIdx] = (
                '{}.'.format(processor.loc[rowIdx]) +
                '{}.'.format(bank.loc[rowIdx]) +
                '{:03d}'.format(int(feBreakout.loc[rowIdx][2]))
                )
        newImpedances['elec'] = FE.map(dict(zip(mapDF['FE'], mapDF['label'])))
        newImpedances['date'] = recDate
        newImpedances['elecType'] = 'isi_paddle'
        newImpedances.dropna(inplace=True)
        saveImpedances = newImpedances.loc[
            :, ['impedance', 'elec', 'elecType', 'date']]
        impList.append(saveImpedances)
    allImpedances = pd.concat(impList, sort=True)
    allImpedances.to_hdf('./impedances.h5', 'impedance')