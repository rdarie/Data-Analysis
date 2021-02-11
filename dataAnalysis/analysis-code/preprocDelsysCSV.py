"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --blockIdx=blockIdx              which trial to analyze
    --exp=exp                        which experimental day to analyze
    --plotting                       show plots? [default: False]
    --verbose                        show plots? [default: False]
    --chanQuery=chanQuery            how to restrict channels if not providing a list?
"""

from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
#
if arguments['plotting']:
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

from tqdm import tqdm
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
from namedQueries import namedQueries
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)
import glob
import line_profiler
import atexit


delsysPath = os.path.join(
    nspFolder, ns5FileName + '.csv')
if not os.path.exists(delsysPath):
    searchStr = os.path.join(nspFolder, '*' + ns5FileName + '*.csv')
    altSearchStr = os.path.join(nspFolder, '*' + 'Block{:0>4}'.format(blockIdx) + '*.csv')
    delsysPathCandidates = glob.glob(searchStr) + glob.glob(altSearchStr)
    assert len(delsysPathCandidates) == 1
    delsysPath = delsysPathCandidates[0]


def preprocDelsysWrapper():
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
    domainCols = [cName for cName in rawData.columns if 'X[' in cName]
    featureCols = [cName for cName in rawData.columns if 'X[' not in cName]
    collatedDataList = []
    print('Assembling list of vectors...')
    for idx, (dom, feat) in enumerate(tqdm(iter(zip(domainCols, featureCols)))):
        newFeat = rawData[feat].to_numpy()
        keepDataMask = rawData[feat].notna()
        newIndex = rawData[dom].interpolate(method='linear')[keepDataMask]
        duplIndex = newIndex.duplicated()
        thisFeat = pd.DataFrame(
            newFeat[keepDataMask][~duplIndex],
            index=newIndex[~duplIndex],
            columns=[feat])
        if idx == 0:
            runningT = [thisFeat.index[0], thisFeat.index[-1]]
        else:
            runningT[0] = min(runningT[0], thisFeat.index[0])
            runningT[-1] = max(runningT[-1], thisFeat.index[-1])
        collatedDataList.append(thisFeat)
    resampledT = np.arange(runningT[0], runningT[-1], samplingRate ** (-1))
    # 
    featureNames = pd.concat([
        df.columns.to_series()
        for df in collatedDataList])
    if arguments['chanQuery'] is not None:
        if arguments['chanQuery'] in namedQueries['chan']:
            chanQuery = namedQueries['chan'][arguments['chanQuery']]
        else:
            chanQuery = arguments['chanQuery']
        chanQuery = chanQuery.replace('chanName', 'featureNames').replace('Emg', 'EMG')
        # pdb.set_trace()
        featureNames = featureNames[eval(chanQuery)]
        collatedDataList = [
            df
            for df in collatedDataList
            if featureNames.str.contains(df.columns[0]).any()]
    print('interpolating...')
    for idx, thisFeat in enumerate(tqdm(collatedDataList)):
        tempT = np.unique(np.concatenate([resampledT, thisFeat.index.to_numpy()]))
        collatedDataList[idx] = (
            thisFeat.reindex(tempT)
            .interpolate(method='linear')
            .fillna(method='ffill').fillna(method='bfill'))
        absentInNew = ~collatedDataList[idx].index.isin(resampledT)
        collatedDataList[idx].drop(
            index=collatedDataList[idx].index[absentInNew],
            inplace=True)
    print('Concatenating...')
    collatedData = pd.concat(collatedDataList, axis=1)
    # TODO: fill zeros with linear interp
    collatedData.columns = [
        re.sub('[\s+]', '', re.sub(r'[^a-zA-Z]', ' ', colName).title())
        for colName in collatedData.columns
        ]
    collatedData.rename(columns={'TrignoAnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog'}, inplace=True)
    collatedData.rename(columns={'AnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog'}, inplace=True)
    collatedData.fillna(method='bfill', inplace=True)
    collatedData.index.name = 't'
    collatedData.reset_index(inplace=True)
    if arguments['plotting']:
        fig, ax = plt.subplots()
        pNames = [
            'AnalogInputAdapterAnalog',
            'RVastusLateralisEmg',
            'RSemitendinosusEmg', 'RPeroneusLongusEmg']
        for cName in pNames:
            plt.plot(
                collatedData['t'],
                collatedData[cName] / collatedData[cName].abs().max(),
                '.-')
        plt.show()
    dataBlock = ns5.dataFrameToAnalogSignals(
        collatedData,
        idxT='t', useColNames=True, probeName='',
        dataCol=collatedData.drop(columns='t').columns,
        samplingRate=samplingRate * pq.Hz, verbose=arguments['verbose'])
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
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=preprocDelsysWrapper,
            modulesToProfile=[ns5],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix, outputUnits=1e-3)
    else:
        preprocDelsysWrapper()
