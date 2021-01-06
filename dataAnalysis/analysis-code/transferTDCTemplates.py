"""03: Transfer curated templates
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx                         which trial to analyze [default: 1]
    --exp=exp                                   which experimental day to analyze
    --processAll                                process entire experimental day? [default: False]
    --fromNS5                                   make from raw ns5 file? [default: False]
    --chan_start=chan_start                     which chan_grp to start on [default: 0]
    --chan_stop=chan_stop                       which chan_grp to stop on [default: 47]
    --arrayName=arrayName                       which electrode array to analyze [default: utah]
"""

import os, pdb, glob, shutil
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import traceback
from currentExperiment import parseAnalysisOptions
from docopt import docopt
import json
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

def transferTDCTemplates():
    # affects the prefix
    arrayName = arguments['arrayName']
    if 'rawBlockName' in spikeSortingOpts[arrayName]:
       rawBlockName = spikeSortingOpts[arrayName]['rawBlockName']
    else:
        rawBlockName = 'Block'
    #
    triFolderSourceMeta = spikeSortingOpts[arrayName]['triFolderSource']
    triFolderSource = os.path.join(
        scratchPath, triFolderSourceMeta['exp'],
        'tdc_{}{:0>3}'.format(rawBlockName, triFolderSourceMeta['block']))
    if triFolderSourceMeta['nameSuffix'] is not None:
        triFolderSource += '_{}'.format(triFolderSourceMeta['nameSuffix'])
    #
    prbPathCandidates = glob.glob(os.path.join(triFolderSource, '*.prb'))
    try:
        assert len(prbPathCandidates) == 1
    except Exception:
        pdb.set_trace()
    prbPath = prbPathCandidates[0]
    #
    dataio = tdc.DataIO(dirname=triFolderSource)
    chan_start = int(arguments['chan_start'])
    chan_stop = int(arguments['chan_stop'])
    chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]
    print('Source TDC catalogue')
    print(dataio)
    #
    for triFolderDestMeta in spikeSortingOpts[arrayName]['triFolderDest']:
        if 'rawBlockName' in spikeSortingOpts[arrayName]:
            blockBaseName = '{}{:0>3}'.format(
                spikeSortingOpts[arrayName]['rawBlockName'],
                triFolderDestMeta['block'])
        else:
            blockBaseName = 'Block{:0>3}'.format(triFolderDestMeta['block'])
        #
        if triFolderDestMeta['nameSuffix'] is not None:
            destNameSuffix = '_{}'.format(triFolderDestMeta['nameSuffix'])
        else:
            destNameSuffix = ''
        #
        chunkingInfoPath = os.path.join(
            scratchPath, triFolderDestMeta['exp'],
            blockBaseName + destNameSuffix + '_chunkingInfo.json'
            )
        pdb.set_trace()
        #
        if os.path.exists(chunkingInfoPath):
            with open(chunkingInfoPath, 'r') as f:
                chunkingMetadata = json.load(f)
        else:
            chunkingMetadata = {
                '0': {
                    'filename': os.path.join(
                        (
                            scratchPath, triFolderDestMeta['exp'],
                            blockBaseName + destNameSuffix + '.nix')
                        ),
                    'partNameSuffix': '',
                    'chunkTStart': 0,
                    'chunkTStop': 'NaN'
                }}
        for chunkIdxStr, chunkMeta in chunkingMetadata.items():
            # chunkIdx = int(chunkIdxStr)
            destNameSuffix = destNameSuffix + chunkMeta['partNameSuffix']
            triFolderDest = os.path.join(
                scratchPath, triFolderDestMeta['exp'],
                'tdc_{}'.format(blockBaseName + destNameSuffix))
            #  #### hack to restore broken templates
            # pdb.set_trace()
            # if True:
            #     # reverse the order
            #     hackSrc = triFolderDest
            #     hackDest = triFolderSource
            #     # tdch.purgeNeoBlock(hackDest)
            #     # tdch.purgePeelerResults(hackDest, purgeAll=True)
            #     tdch.transferTemplates(hackSrc, hackDest, chansToAnalyze)
            #     break
            ##################
            if os.path.exists(triFolderDest):
                shutil.rmtree(triFolderDest)
            try:
                if arguments['fromNS5']:
                    tdch.initialize_catalogueconstructor(
                        remoteBasePath, 'raw', triFolderDestMeta['exp'],
                        blockBaseName,
                        triFolderDest,
                        prbPath=prbPath,
                        fileFormat='Blackrock')
                else:
                    tdch.initialize_catalogueconstructor(
                        scratchPath, triFolderDestMeta['exp'],
                        blockBaseName,
                        triFolderDest, prbPath=prbPath,
                        spikeSortingOpts=spikeSortingOpts[arrayName],
                        fileFormat='NIX')
            except Exception:
                traceback.print_exc()
                print('Ignoring Exception')
            allCatPaths = glob.glob(os.path.join(triFolderSource, '*', 'catalogues'))
            print('List of all catalogues in the source: ')
            for pa in sorted(allCatPaths):
                print('        {}'.format(pa))
            tdch.transferTemplates(
                triFolderSource, triFolderDest,
                chansToAnalyze)
    return


if __name__ == "__main__":
    runProfiler = False
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        taskNameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=transferTDCTemplates,
            modulesToProfile=[tdch],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=taskNameSuffix, outputUnits=1e-3)
    else:
        transferTDCTemplates()
    print('Done running templateTransfer.py')