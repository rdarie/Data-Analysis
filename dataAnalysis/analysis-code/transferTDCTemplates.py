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

import os, pdb, glob
import tridesclous as tdc
import dataAnalysis.helperFunctions.tridesclous_helpers as tdch
import traceback
from currentExperiment import parseAnalysisOptions
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']),
    arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

arrayName = arguments['arrayName']
if 'rawBlockName' in spikeSortingOpts[arrayName]:
   rawBlockName = spikeSortingOpts[arrayName]['rawBlockName']
else:
    rawBlockName = 'Block'
triFolderSourceMeta = spikeSortingOpts[arrayName]['triFolderSource']
triFolderSource = os.path.join(
    scratchPath, triFolderSourceMeta['exp'],
    'tdc_{}{:0>3}'.format(rawBlockName, triFolderSourceMeta['block']))
if triFolderSourceMeta['nameSuffix'] is not None:
    triFolderSource += '_{}'.format(triFolderSourceMeta['nameSuffix'])
#
prbPathCandidates = glob.glob(os.path.join(triFolderSource, '*.prb'))
assert len(prbPathCandidates) == 1
prbPath = prbPathCandidates[0]
#
dataio = tdc.DataIO(dirname=triFolderSource)
chan_start = int(arguments['chan_start'])
chan_stop = int(arguments['chan_stop'])
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]

for triFolderDestMeta in spikeSortingOpts[arrayName]['triFolderDest']:
    triFolderDest = os.path.join(
        scratchPath, triFolderDestMeta['exp'],
        'tdc_{}{:0>3}'.format(rawBlockName, triFolderDestMeta['block']))
    if 'rawBlockName' in spikeSortingOpts[arrayName]:
        ns5FileName = '{}{:0>3}'.format(spikeSortingOpts[arrayName]['rawBlockName'], triFolderDestMeta['block'])
    else:
        ns5FileName = 'Block{:0>3}'.format(triFolderDestMeta['block'])
    if triFolderDestMeta['nameSuffix'] is not None:
        triFolderDest += '_{}'.format(triFolderDestMeta['nameSuffix'])
        ns5FileName += '_{}'.format(triFolderDestMeta['nameSuffix'])
    ##### hack to restore broken templates
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
    try:
        if arguments['fromNS5']:
            tdch.initialize_catalogueconstructor(
                nspFolder,
                ns5FileName,
                triFolderDest,
                prbPath=prbPath,
                fileFormat='Blackrock')
        else:
            tdch.initialize_catalogueconstructor(
                scratchFolder,
                ns5FileName,
                triFolderDest, prbPath=prbPath,
                spikeSortingOpts=spikeSortingOpts[arrayName],
                fileFormat='NIX')
    except Exception:
        traceback.print_exc()
        print('Ignoring Exception')
    tdch.transferTemplates(
        triFolderSource, triFolderDest,
        chansToAnalyze)
