"""06a: Preprocess the NS5 File

Usage:
    preprocNS5.py [options]

Options:
    --blockIdx=blockIdx                which trial to analyze
    --exp=exp                          which experimental day to analyze
    --arrayName=arrayName              use a named array? [default: Block]
    --chunkSize=chunkSize              split into blocks of this size
    --analogOnly                       whether to make a .nix file that has all raw traces [default: False]
    --forSpikeSorting                  whether to make a .nix file that has all raw traces [default: False]
    --forSpikeSortingUnfiltered        whether to make a .nix file that has all raw traces [default: False]
    --fullSubtractMean                 whether to make a .nix file that has all raw traces [default: False]
    --fullSubtractMeanWithSpikes       whether to make a .nix file that has all raw traces [default: False]
    --fullSubtractMeanUnfiltered       whether to make a .nix file that has all raw traces [default: False]
    --rippleNForm                      whether to make a .nix file that has all raw traces [default: False]
    --makeFull                         whether to make a .nix file that has all raw traces [default: False]
    --maskMotorEncoder                 whether to ignore motor encoder activity outside the alignTimeBounds window [default: False]
    --ISI                              special options for parsing Ripple files from ISI [default: False]
    --transferISIStimLog               special options for parsing Ripple files from ISI [default: False]
    --ISIMinimal                       special options for parsing Ripple files from ISI [default: False]
    --ISIRaw                           special options for parsing Ripple files from ISI [default: False]
    
"""

import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import pdb, traceback, shutil, os

#  load options
from currentExperiment import parseAnalysisOptions
from docopt import docopt
#
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)


def preprocNS5():
    # weird scope issue with ns5FileName in particular
    ns5FileName = allOpts['ns5FileName']
    arrayName = arguments['arrayName']
    if arguments['arrayName'] != 'Block':
        electrodeMapPath = spikeSortingOpts[arrayName]['electrodeMapPath']
        mapExt = electrodeMapPath.split('.')[-1]
        if mapExt == 'cmp':
            mapDF = prb_meta.cmpToDF(electrodeMapPath)
        elif mapExt == 'map':
            mapDF = prb_meta.mapToDF(electrodeMapPath)
        if 'rawBlockName' in spikeSortingOpts[arrayName]:
            ns5FileName = ns5FileName.replace(
                'Block', spikeSortingOpts[arrayName]['rawBlockName'])
    idealDataPath = os.path.join(nspFolder, ns5FileName + '.ns5')
    if not os.path.exists(idealDataPath):
        fallBackPath = os.path.join(
            nspFolder,
            '{}{:0>4}'.format(arrayName, blockIdx) + '.ns5')
        print('{} not found;\nFalling back to {}'.format(
            idealDataPath, fallBackPath
        ))
        if os.path.exists(fallBackPath):
            shutil.move(
                fallBackPath,
                idealDataPath)
            try:
                shutil.move(
                    fallBackPath.replace('.ns5', '.nev'),
                    idealDataPath.replace('.ns5', '.nev'))
            except Exception:
                traceback.print_exc()
                print('Ignoring exception...')

    if arguments['chunkSize'] is not None:
        chunkSize = int(arguments['chunkSize'])
    else:
        chunkSize = 4000
    chunkList = None
    equalChunks = False
    ###############################################################
    groupAsigsByBank = True
    if groupAsigsByBank:
        print('Rewriting list of asigs that will be processed')
        spikeSortingOpts[arrayName]['asigNameList'] = []
        for name, group in mapDF.groupby('bank'):
            allAsigsInBank = sorted(group['label'].to_list())
            theseAsigNames = [
                aName
                for aName in allAsigsInBank
                if aName not in spikeSortingOpts[arrayName]['excludeChans']
                ]
            spikeSortingOpts[arrayName]['asigNameList'].append(theseAsigNames)
            print(theseAsigNames)
    ###############################################################
    if arguments['maskMotorEncoder']:
        motorEncoderMask = alignTimeBoundsLookup[int(arguments['blockIdx'])]
    else:
        motorEncoderMask = None
    ###############################################################
    #
    if arguments['rippleNForm']:
        analogInputNames = sorted(
            trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
        # pdb.set_trace()
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False, electrodeArrayName=arrayName,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True, removeMeanAcross=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=spikeSortingOpts[arrayName]['ainpNameList'],
            spikeSourceType='tdc', writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    #
    if arguments['forSpikeSorting']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=[],
            spikeSourceType='',
            removeMeanAcross=True,
            nameSuffix='_spike_preview',
            LFPFilterOpts=spikeSortingFilterOpts,
            # LFPFilterOpts=None,
            writeMode='ow',
            chunkSize=spikeSortingOpts[arrayName]['previewDuration'],
            chunkOffset=spikeSortingOpts[arrayName]['previewOffset'],
            equalChunks=False, chunkList=[0],
            calcRigEvents=False)
    #
    if arguments['forSpikeSortingUnfiltered']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            # interpolateOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            interpolateOutliers=False,
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=spikeSortingOpts[arrayName]['ainpNameList'],
            spikeSourceType='',
            removeMeanAcross=False,
            nameSuffix='_spike_preview_unfiltered',
            # LFPFilterOpts=spikeSortingFilterOpts,
            LFPFilterOpts=None,
            writeMode='ow',
            chunkSize=spikeSortingOpts[arrayName]['previewDuration'],
            chunkOffset=spikeSortingOpts[arrayName]['previewOffset'],
            equalChunks=False, chunkList=[0],
            calcRigEvents=False)
    #
    #
    if arguments['fullSubtractMean']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=[],
            spikeSourceType='',
            removeMeanAcross=True,
            nameSuffix='_mean_subtracted',
            LFPFilterOpts=spikeSortingFilterOpts,
            #
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    if arguments['fullSubtractMeanUnfiltered']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=[],
            spikeSourceType='',
            removeMeanAcross=True,
            nameSuffix='',
            LFPFilterOpts=None,
            #
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    #
    if arguments['analogOnly']:
        analogInputNames = sorted(
            trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=[],
            ainpNameList=analogInputNames,
            spikeSourceType='',
            nameSuffix='_analog_inputs', writeMode='ow',
            chunkSize=4000,
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'])
    #
    if arguments['fullSubtractMeanWithSpikes']:
        spikePath = os.path.join(
            scratchFolder, 'tdc_' + ns5FileName + '_mean_subtracted',
            'tdc_' + ns5FileName + '_mean_subtracted' + '.nix'
            )
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            # swapMaps=None,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            ainpNameList=[],
            removeMeanAcross=True,
            LFPFilterOpts=None,
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            nameSuffix='',
            spikeSourceType='tdc', spikePath=spikePath,
            #
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'])
    ###############################################################################
    if arguments['ISI'] or arguments['ISIRaw'] or arguments['ISIMinimal']:
        mapDF = prb_meta.mapToDF(rippleMapFile[int(arguments['blockIdx'])])
        # if 'rippleOriginalMapFile' in expOpts:
        #     rippleOriginalMapFile = expOpts['rippleOriginalMapFile']
        #     if rippleOriginalMapFile[int(arguments['blockIdx'])] is not None:
        #         swapMaps = {
        #             'from': prb_meta.mapToDF(rippleOriginalMapFile[int(arguments['blockIdx'])]),
        #             'to': mapDF
        #         }
        #     else:
        #         swapMaps = None
        # else:
        #     swapMaps = None
    if arguments['ISI']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            # swapMaps=swapMaps,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            spikeSourceType='nev', writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks,
            chunkList=chunkList,
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'],
            normalizeByImpedance=False, removeMeanAcross=False,
            asigNameList=asigNameList, ainpNameList=ainpNameList,
            # LFPFilterOpts=LFPFilterOpts,
            LFPFilterOpts=None,
            calcAverageLFP=True)
        if arguments['transferISIStimLog']:
            try:
                jsonSrcPath = os.path.join(nspFolder, ns5FileName + '_autoStimLog.json')
                jsonDestPath = trialBasePath.replace('.nix', '_autoStimLog.json')
                shutil.copyfile(jsonSrcPath, jsonDestPath)
            except Exception:
                traceback.print_exc()
    if arguments['ISIMinimal']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder,
            mapDF=mapDF,
            #swapMaps=swapMaps,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            spikeSourceType='nev', writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks,
            chunkList=chunkList,
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'],
            normalizeByImpedance=False, removeMeanAcross=False,
            asigNameList=[], ainpNameList=ainpNameList,
            # LFPFilterOpts=LFPFilterOpts,
            LFPFilterOpts=None, calcAverageLFP=False)
        if arguments['transferISIStimLog']:
            try:
                jsonSrcPath = os.path.join(nspFolder, ns5FileName + '_autoStimLog.json')
                jsonDestPath = trialBasePath.replace('.nix', '_autoStimLog.json')
                shutil.copyfile(jsonSrcPath, jsonDestPath)
            except Exception:
                traceback.print_exc()
    ##################################################################################
    if arguments['ISIRaw']:
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=None,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            spikeSourceType='nev', writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks,
            chunkList=chunkList,
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'],
            normalizeByImpedance=False, removeMeanAcross=False,
            asigNameList=None, ainpNameList=None, nameSuffix='_raw',
            LFPFilterOpts=LFPFilterOpts, calcAverageLFP=True)
    return


if __name__ == "__main__":
    runProfiler = True
    if runProfiler:
        import dataAnalysis.helperFunctions.profiling as prf
        nameSuffix = os.environ.get('SLURM_ARRAY_TASK_ID')
        prf.profileFunction(
            topFun=preprocNS5,
            modulesToProfile=[ns5],
            outputBaseFolder=os.path.join(remoteBasePath, 'batch_logs'),
            nameSuffix=nameSuffix, outputUnits=1e-3)
    else:
        preprocNS5()
    print('Done running preprocNS5.py')
