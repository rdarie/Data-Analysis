"""06b: Preprocess the INS Recording

Usage:
    preprocINSData.py [options]

Options:
    --blockIdx=blockIdx              which trial to analyze
    --exp=exp                        which experimental day to analyze
    --plotting                       show plots? [default: False]
    --chanQuery=chanQuery            how to restrict channels if not providing a list? [default: fr]
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
import dataAnalysis.helperFunctions.helper_functions_new as hf
import os, glob, shutil
import pdb, traceback
import re
from importlib import reload
#  load options
from currentExperiment import parseAnalysisOptions
from namedQueries import namedQueries
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

import line_profiler
import atexit
#profile = line_profiler.LineProfiler()
#atexit.register(profile.print_stats)

import btk
interpKind = 'linear'

def preprocDelsysWrapper():
    featureRenameLookup = {
        'TrignoAnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog',
        'AnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog'}
    print('Loading {}'.format(delsysExampleHeaderPath))
    headerDataList = []
    with open(delsysExampleHeaderPath, 'r') as f:
        expr = r'Label: ([\S\s]+) Sampling frequency: ([\S\s]+) Number of points: ([\S\s]+) start: ([\S\s]+) Unit: ([\S\s]+) Domain Unit: ([\S\s]+)\n'
        delimIdx = 0
        for line in f:
            matches = re.search(expr, line)
            if matches:
                thisFs = float(matches.groups()[1])
                # thisNSamp = int(matches.groups()[2])
                # thisStart = float(matches.groups()[3])
                # thisT = thisStart + (thisFs ** -1) * np.arange(thisNSamp)
                headerDataList.append({
                    'label': str(matches.groups()[0]),
                    'fs': thisFs,
                    # 'nSamp': thisNSamp,
                    # 'start': thisStart,
                    'units': str(matches.groups()[4]),
                    'domainUnits': str(matches.groups()[5]),
                    # 't': thisT
                })
            elif line == ' \n':
                break
            delimIdx += 1
    # pdb.set_trace()
    headerData = pd.DataFrame(headerDataList).set_index('label')
    # samplingRate = np.round(headerData['fs'].max())
    fastestChanName = headerData['fs'].idxmax()
    samplingRate = headerData.loc[fastestChanName, 'fs']
    # referenceT = headerData.loc[fastestChanName, 't']
    #
    searchStr = os.path.join(nspFolder, '*' + ns5FileName + '*.hpf')
    delsysPathCandidates = glob.glob(searchStr)
    assert len(delsysPathCandidates) == 1
    delsysPath = delsysPathCandidates[0]
    #
    delsysPathShort = os.path.join(nspFolder, ns5FileName + '.hpf')
    if delsysPathShort != delsysPath:
        shutil.move(delsysPath, delsysPathShort)
        delsysPath = delsysPathShort
    #
    if arguments['chanQuery'] is not None:
        if arguments['chanQuery'] in namedQueries['chan']:
            chanQuery = namedQueries['chan'][arguments['chanQuery']]
        else:
            chanQuery = arguments['chanQuery']
    #   chanQuery = chanQuery.replace('.str', '').replace('contains', 'find')
    reader = btk.btkAcquisitionFileReader()  # build a btk reader object
    reader.SetFilename(delsysPath)  # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput()  # acq is the btk aquisition object
    data = {}
    metaData = {}
    #
    dummyAnalog = acq.GetAnalog(0)
    nSampMax = dummyAnalog.GetFrameNumber()
    referenceT = samplingRate ** -1 * np.arange(nSampMax)
    for idx in range(acq.GetAnalogNumber()):
        analog = acq.GetAnalog(idx)
        thisLabel = analog.GetLabel()
        featName = re.sub('[\s+]', '', re.sub(r'[^a-zA-Z]', ' ', thisLabel).title())
        if featName in featureRenameLookup:
            featName = featureRenameLookup[featName]
        chanName = pd.Series(featName)
        # pdb.set_trace()
        if not eval(chanQuery)[0]:
            print('Not loading {} bc. of query'.format(thisLabel))
            continue
        else:
            print('Loading {}'.format(thisLabel))
        # thisNSamp = headerData.loc[thisLabel, 'nSamp']
        thisFeat = pd.Series(analog.GetValues().flatten()).to_frame(name=featName)
        thisFs = headerData.loc[thisLabel, 'fs']
        if thisFs != samplingRate:
            thisNSamp = np.round(nSampMax * thisFs / samplingRate).astype(np.int)
            thisFeat = thisFeat.iloc[:thisNSamp]
            thisFeat.index = thisFs ** -1 * np.arange(thisNSamp)
            thisFeat = hf.interpolateDF(
                thisFeat, referenceT, kind=interpKind)
        data[featName] = thisFeat.to_numpy().flatten()
        metaData[featName] = {
            'gain': analog.GetGain(),
            'offset': analog.GetOffset(),
            'scale': analog.GetScale(),
            'timestamp': analog.GetTimestamp(),
            'unit': analog.GetUnit()
            }
    pdb.set_trace()
    dataDF = pd.DataFrame(data)
    # plt.plot(data['L THORACOLUMBAR FASCIA 12: Acc 14.Y'])
    # plt.plot(data['L THORACOLUMBAR FASCIA 12: EMG 14'])
    # plt.plot(data['Trigno Analog Input Adapter 16: Analog 16.A'])
    #  resampledT = np.arange(runningT[0], runningT[-1], samplingRate ** (-1))
    #  # 
    #  featureNames = pd.concat([
    #      df.columns.to_series()
    #      for df in collatedDataList])
    #  if arguments['chanQuery'] is not None:
    #      if arguments['chanQuery'] in namedQueries['chan']:
    #          chanQuery = namedQueries['chan'][arguments['chanQuery']]
    #      else:
    #          chanQuery = arguments['chanQuery']
    #      chanQuery = chanQuery.replace('chanName', 'featureNames').replace('Emg', 'EMG')
    #      # pdb.set_trace()
    #      featureNames = featureNames[eval(chanQuery)]
    #      collatedDataList = [
    #          df
    #          for df in collatedDataList
    #          if featureNames.str.contains(df.columns[0]).any()]
    #  print('interpolating...')
    #  for idx, thisFeat in enumerate(tqdm(collatedDataList)):
    #      tempT = np.unique(np.concatenate([resampledT, thisFeat.index.to_numpy()]))
    #      collatedDataList[idx] = (
    #          thisFeat.reindex(tempT)
    #          .interpolate(method='pchip')
    #          .fillna(method='ffill').fillna(method='bfill'))
    #      absentInNew = ~collatedDataList[idx].index.isin(resampledT)
    #      collatedDataList[idx].drop(
    #          index=collatedDataList[idx].index[absentInNew],
    #          inplace=True)
    #  print('Concatenating...')
    #  collatedData = pd.concat(collatedDataList, axis=1)
    #  collatedData.columns = [
    #      re.sub('[\s+]', '', re.sub(r'[^a-zA-Z]', ' ', colName).title())
    #      for colName in collatedData.columns
    #      ]
    #  collatedData.rename(columns={'TrignoAnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog'}, inplace=True)
    #  collatedData.rename(columns={'AnalogInputAdapterAnalogA': 'AnalogInputAdapterAnalog'}, inplace=True)
    #  collatedData.fillna(method='bfill', inplace=True)
    #  collatedData.index.name = 't'
    #  collatedData.reset_index(inplace=True)
    #  if arguments['plotting']:
    #      fig, ax = plt.subplots()
    #      pNames = [
    #          'AnalogInputAdapterAnalog',
    #          'RVastusLateralisEmg',
    #          'RSemitendinosusEmg', 'RPeroneusLongusEmg']
    #      for cName in pNames:
    #          plt.plot(
    #              collatedData['t'],
    #              collatedData[cName] / collatedData[cName].abs().max(),
    #              '.-')
    #      plt.show()
    dataBlock = ns5.dataFrameToAnalogSignals(
        collatedData,
        idxT='t', useColNames=True, probeName='',
        dataCol=collatedData.drop(columns='t').columns,
        samplingRate=samplingRate * pq.Hz)
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
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=preprocDelsysWrapper,
            modulesToProfile=[ns5],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix)
    else:
        preprocDelsysWrapper()
