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

def preprocDelsysWrapper():
    searchStr = os.path.join(nspFolder, '*' + ns5FileName + '*.hpf')

    delsysPathCandidates = glob.glob(searchStr)
    assert len(delsysPathCandidates) == 1
    delsysPath = delsysPathCandidates[0]
    delsysPathShort = os.path.join(nspFolder, ns5FileName + '.hpf')
    shutil.move(delsysPath, delsysPathShort)
    delsysPath = delsysPathShort
    pdb.set_trace()
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(delsysPath) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    bla = acq.GetAnalog(0)
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
            .interpolate(method='pchip')
            .fillna(method='ffill').fillna(method='bfill'))
        absentInNew = ~collatedDataList[idx].index.isin(resampledT)
        collatedDataList[idx].drop(
            index=collatedDataList[idx].index[absentInNew],
            inplace=True)
    print('Concatenating...')
    collatedData = pd.concat(collatedDataList, axis=1)
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
    runProfiler = True
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
