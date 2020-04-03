"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --blockIdx=blockIdx              which trial to analyze
    --exp=exp                        which experimental day to analyze
    --plotting                       show plots? [default: False]
    --disableStimDetection           disable stimulation time detection? [default: False]
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive qt output
# matplotlib.use('PS')   # generate offline postscript
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("white")

from neo.io import NixIO, nixio_fr, BlackrockIO
import pandas as pd
import numpy as np
import quantities as pq
import dataAnalysis.preproc.ns5 as ns5
import os
import pdb, traceback
import re
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

import line_profiler
import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

def preprocDelsysWrapper(
        delsysPath=None,
        arguments=None
        ):
    headerDataList = []
    with open(delsysPath, 'r') as f:
        expr = r'Label: ([\S\s]+) Sampling frequency: ([\S\s]+) Number of points: ([\S\s]+) start: ([\S\s]+) Unit: ([\S\s]+) Domain Unit: ([\S\s]+)\n'
        delimIdx = 0
        for line in f:
            matches = re.search(expr, line)
            if matches:
                headerDataList.append({
                    'label': str(matches.groups()[0]),
                    'fs': float(matches.groups()[1]),
                    'nSamp': int(matches.groups()[2]),
                    'start': float(matches.groups()[3]),
                    'units': str(matches.groups()[4]),
                    'domainUnits': str(matches.groups()[5])
                })
            elif line == ' \n':
                break
            delimIdx += 1
    headerData = pd.DataFrame(headerDataList)
    samplingRate = np.round(headerData['fs'].max())
    #
    rawData = pd.read_csv(delsysPath, skiprows=delimIdx, low_memory=False)
    
    # for idx, cName in enumerate(rawData.columns): print('{}: {}'.format(idx, cName))
    if arguments['plotting']:
        ainp = rawData['Analog Input adapter 7: Analog 7']
        plt.plot(rawData['X[s].42'], ainp / ainp.abs().max(), '.-')
        trace = rawData.iloc[:, 1]
        plt.plot(rawData.iloc[:, 0], trace / trace.abs().max(), '.-')
        trace = rawData.iloc[:, 15]
        plt.plot(rawData.iloc[:, 14], trace / trace.abs().max(), '.-')
        trace = rawData.iloc[:, 29]
        plt.plot(rawData.iloc[:, 28], trace / trace.abs().max(), '.-')
        plt.show(block=False)
    domainCols = [cName for cName in rawData.columns if 'X[' in cName]
    featureCols = [cName for cName in rawData.columns if 'X[' not in cName]
    collatedDataList = []
    for idx, (dom, feat) in enumerate(zip(domainCols, featureCols)):
        newIndex = rawData[dom].to_numpy()
        newFeat = rawData[feat].to_numpy()
        keepDataMask = (
            rawData[dom].notna() & rawData[feat].notna())
        thisFeat = pd.DataFrame(
            newFeat[keepDataMask],
            index=newIndex[keepDataMask],
            columns=[feat]
            )
        thisFeat.drop(
            index=thisFeat.index[thisFeat.index.duplicated()],
            inplace=True)
        if idx == 0:
            runningT = thisFeat.index.to_numpy()
        else:
            runningT = np.concatenate([runningT, thisFeat.index.to_numpy()])
        if rawData[dom].isna().any():
            print('{} NaNs detected: '.format(rawData[dom].isna().sum()))
            print(feat)
            print(np.flatnonzero(rawData[dom].isna()))
        collatedDataList.append(thisFeat)
    #
    finalT = np.unique(runningT)
    resampledT = np.arange(finalT[0], finalT[-1], samplingRate ** (-1))
    for idx, thisFeat in enumerate(collatedDataList):
        tempT = np.unique(np.concatenate([resampledT, thisFeat.index.to_numpy()]))
        collatedDataList[idx] = (
            thisFeat.reindex(tempT)
            .interpolate(method='cubic')
            .fillna(method='ffill').fillna(method='bfill'))
        absentInNew = ~collatedDataList[idx].index.isin(resampledT)
        collatedDataList[idx].drop(
            index=collatedDataList[idx].index[absentInNew],
            inplace=True)
    collatedData = pd.concat(collatedDataList, axis=1)
    collatedData.columns = [
        re.sub('[\s+]', '', re.sub(r'[^a-zA-Z]', ' ', colName).title())
        for colName in collatedData.columns
        ]
    #
    collatedData.fillna(method='bfill', inplace=True)
    collatedData.index.name = 't'
    collatedData.reset_index(inplace=True)
    # pdb.set_trace()
    if arguments['plotting']:
        fig, ax = plt.subplots()
        pNames = [
            'AnalogInputAdapterAnalog',
            'RVastusLateralisEmg',
            'RSemitendinosusEmg', 'RPeroneusLongusEmg']
        for cName in pNames:
            plt.plot(
                collatedData['t'],
                collatedData[cName] / collatedData[cName].abs().max(), '.-')
        plt.show()
    dataBlock = ns5.dataFrameToAnalogSignals(
        collatedData,
        idxT='t', useColNames=True, probeName='',
        dataCol=collatedData.drop(columns='t').columns,
        samplingRate=samplingRate * pq.Hz)
    # pdb.set_trace()
    dataBlock.name = 'delsys'
    outPathName = os.path.join(
        scratchFolder, ns5FileName + '_delsys.nix')
    if os.path.exists(outPathName):
        os.remove(outPathName)
    writer = NixIO(filename=outPathName)
    writer.write_block(dataBlock, use_obj_names=True)
    writer.close()
    return


if __name__ == "__main__":
    delsysPath = os.path.join(
        nspFolder, ns5FileName + '.csv')
    preprocDelsysWrapper(
        delsysPath=delsysPath,
        arguments=arguments)
