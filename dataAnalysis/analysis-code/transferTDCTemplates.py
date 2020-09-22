"""03: Transfer curated templates
Usage:
    temp.py [options]

Options:
    --blockIdx=blockIdx             which trial to analyze [default: 1]
    --exp=exp                       which experimental day to analyze
    --processAll                    process entire experimental day? [default: False]
    --sourceFile=sourceFile         which source file to analyze [default: raw]
    --chan_start=chan_start         which chan_grp to start on [default: 0]
    --chan_stop=chan_stop           which chan_grp to stop on [default: 47]
    --arrayName=arrayName           which electrode array to analyze [default: utah]
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
spikeSortingOpts = spikeSortingOpts[arrayName]
triFolderSourceMeta = spikeSortingOpts['triFolderSource']
triFolderSource = os.path.join(
    scratchPath, triFolderSourceMeta['exp'],
    'tdc_{}{:0>3}'.format(arrayName, triFolderSourceMeta['block']))
#
ns5FileName = ns5FileName.replace('Block', arrayName)
prbPathCandidates = glob.glob(os.path.join(triFolderSource, '*.prb'))
assert len(prbPathCandidates) == 1
prbPath = prbPathCandidates[0]
#
dataio = tdc.DataIO(dirname=triFolderSource)
chan_start = int(arguments['chan_start'])
chan_stop = int(arguments['chan_stop'])
chansToAnalyze = sorted(list(dataio.channel_groups.keys()))[chan_start:chan_stop]

for triFolderDestMeta in spikeSortingOpts['triFolderDest']:
    triFolderDest = os.path.join(
        scratchPath, triFolderDestMeta['exp'],
        'tdc_{}{:0>3}'.format(arrayName, triFolderDestMeta['block']))
    try:
        if arguments['sourceFile'] == 'raw':
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
                spikeSortingOpts=spikeSortingOpts,
                fileFormat='NIX')
    except Exception:
        traceback.print_exc()
        print('Ignoring Exception')
    tdch.transferTemplates(
        triFolderSource, triFolderDest,
        chansToAnalyze)
