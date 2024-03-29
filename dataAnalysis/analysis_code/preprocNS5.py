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
    --fullUnfiltered                   whether to make a .nix file that has all raw traces [default: False]
    --rippleNForm                      whether to make a .nix file that has all raw traces [default: False]
    --makeFull                         whether to make a .nix file that has all raw traces [default: False]
    --maskMotorEncoder                 whether to ignore motor encoder activity outside the alignTimeBounds window [default: False]
    --ISI                              special options for parsing Ripple files from ISI [default: False]
    --transferISIStimLog               special options for parsing Ripple files from ISI [default: False]
    --ISIMinimal                       special options for parsing Ripple files from ISI [default: False]
    --ISIRaw                           special options for parsing Ripple files from ISI [default: False]
    
"""
import logging
logging.captureWarnings(True)
import os, sys

from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from dataAnalysis.analysis_code.namedQueries import namedQueries

########################################################################################################################
## if plotting
########################################################################################################################
import matplotlib
if 'CCV_HEADLESS' in os.environ:
    matplotlib.use('Agg')   # generate postscript output
else:
    matplotlib.use('QT5Agg')   # generate interactive output
import matplotlib.font_manager as fm
font_files = fm.findSystemFonts()
for font_file in font_files:
    try:
        fm.fontManager.addfont(font_file)
    except Exception:
        pass
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
########################################################################################################################
import dataAnalysis.preproc.ns5 as ns5
import dataAnalysis.helperFunctions.probe_metadata as prb_meta
import pdb, traceback, shutil
#  load options
from dataAnalysis.analysis_code.currentExperiment import parseAnalysisOptions
from docopt import docopt
#
from datetime import datetime as dt
try:
    print('\n' + '#' * 50 + '\n{}\n{}\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
except:
    pass
for arg in sys.argv:
    print(arg)
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
        # mapDF = mapDF.loc[mapDF['elecName'] == arrayName, :].reset_index(drop=True)
    idealDataPath = os.path.join(nspFolder, ns5FileName + '.ns5')
    if not os.path.exists(idealDataPath):
        fallBackPathList = [
            os.path.join(
                nspFolder,
                '{}{:0>4}'.format(arrayName, blockIdx) + '.ns5'),
            os.path.join(
                nspFolder,
                '{}{:0>3}'.format('Block', blockIdx) + '.ns5'),
            ]
        for fbp in fallBackPathList:
            if os.path.exists(fbp):
                print('{} not found;\nFalling back to {}'.format(
                    idealDataPath, fbp))
                shutil.move(fbp, idealDataPath)
                for auxExt in ['.nev', '.ccf']:
                    try:
                        shutil.move(
                            fbp.replace('.ns5', auxExt),
                            idealDataPath.replace('.ns5', auxExt))
                        print('Moved \n{} to \n{}'.format(fbp, idealDataPath))
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
    # pdb.set_trace()
    if groupAsigsByBank:
        try:
            print('Rewriting list of asigs that will be processed')
            asigNameListByBank = []
            # spikeSortingOpts[arrayName]['asigNameList'] = []
            for name, group in mapDF.groupby('bank'):
                allAsigsInBank = sorted(group['label'].to_list())
                theseAsigNames = [
                    aName
                    for aName in allAsigsInBank
                    if aName not in spikeSortingOpts[arrayName]['excludeChans']
                    ]
                asigNameListByBank.append(theseAsigNames)
                # spikeSortingOpts[arrayName]['asigNameList'].append(theseAsigNames)
                print(theseAsigNames)
        except Exception:
            asigNameListByBank = None
    ###############################################################
    if arguments['maskMotorEncoder']:
        try:
            motorEncoderMask = motorEncoderBoundsLookup[int(arguments['blockIdx'])]
        except Exception:
            traceback.print_exc()
            try:
                motorEncoderMask = alignTimeBoundsLookup[int(arguments['blockIdx'])]
            except Exception:
                traceback.print_exc()
                motorEncoderMask = None
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
        print('\n\nPreprocNs5, generating spike preview...\n\n')
        if asigNameListByBank is not None:
            theseAsigNames = asigNameListByBank
        else:
            theseAsigNames = spikeSortingOpts[arrayName]['asigNameList']
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            calcArtifactTrace=spikeSortingOpts[arrayName]['interpolateOutliers'],
            calcOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            interpolateOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=theseAsigNames,
            ainpNameList=[],
            spikeSourceType='',
            removeMeanAcross=True,
            linearDetrend=True,
            nameSuffix='_spike_preview',
            LFPFilterOpts=spikeSortingFilterOpts,
            # LFPFilterOpts=None,
            writeMode='ow',
            chunkSize=spikeSortingOpts[arrayName]['previewDuration'],
            chunkOffset=spikeSortingOpts[arrayName]['previewOffset'],
            equalChunks=False, chunkList=[0],
            calcRigEvents=False, outlierRemovalDebugFlag=False)
    #
    if arguments['fullSubtractMean']:
        print('\n\nPreprocNs5, generating spike extraction data...\n\n')
        if asigNameListByBank is not None:
            theseAsigNames = asigNameListByBank
        else:
            theseAsigNames = spikeSortingOpts[arrayName]['asigNameList']
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            calcOutliers=spikeSortingOpts[arrayName]['interpolateOutliers'],
            interpolateOutliers=False,
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True, removeMeanAcross=True, linearDetrend=True,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=theseAsigNames,
            ainpNameList=[],
            spikeSourceType='',
            nameSuffix='_mean_subtracted',
            LFPFilterOpts=spikeSortingFilterOpts,
            #
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    #
    if arguments['fullSubtractMeanUnfiltered']:
        print('\n\nPreprocNs5, generating lfp data...\n\n')
        theseAsigNames = [mapDF['label'].iloc[::10].to_list()]
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=False, calcOutliers=False, calcArtifactTrace=False,
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=True, removeMeanAcross=True, linearDetrend=False,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            impedanceFilePath=os.path.join(
                remoteBasePath,
                '{}_blackrock_impedances.h5'.format(subjectName)),
            ainpNameList=[],
            spikeSourceType='',
            nameSuffix='_CAR',
            LFPFilterOpts=None,
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    #
    if arguments['fullUnfiltered']:
        print('\n\nPreprocNs5, generating lfp data...\n\n')
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=False, calcOutliers=False, calcArtifactTrace=False,
            outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            calcAverageLFP=False, removeMeanAcross=False, linearDetrend=False,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=spikeSortingOpts[arrayName]['asigNameList'],
            normalizeByImpedance=False,
            impedanceFilePath=os.path.join(
                remoteBasePath,
                '{}_blackrock_impedances.h5'.format(subjectName)),
            ainpNameList=[],
            spikeSourceType='',
            nameSuffix='',
            LFPFilterOpts=None,
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            calcRigEvents=False)
    #
    if arguments['analogOnly']:
        analogInputNames = sorted(
            trialFilesFrom['utah']['eventInfo']['inputIDs'].values())
        theseAsigNames = [mapDF.loc[mapDF['elecName'] == arguments['arrayName'], 'label'].iloc[::3].to_list()]
        print('\n\nPreprocNs5, generating rig inputs and other analog data...\n\n')
        ns5.preproc(
            fileName=ns5FileName,
            rawFolderPath=nspFolder,
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            interpolateOutliers=False, calcOutliers=False,
            calcArtifactTrace=True,
            # outlierThreshold=spikeSortingOpts[arrayName]['outlierThreshold'],
            # outlierMaskFilterOpts=outlierMaskFilterOpts,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            asigNameList=theseAsigNames,
            saveFromAsigNameList=False,
            calcAverageLFP=True,
            LFPFilterOpts=stimArtifactFilterOpts,
            ainpNameList=analogInputNames,
            spikeSourceType='',
            nameSuffix='_analog_inputs', writeMode='ow',
            chunkSize=9999,
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
            outputFolderPath=scratchFolder, mapDF=mapDF,
            fillOverflow=False, removeJumps=False,
            motorEncoderMask=motorEncoderMask,
            eventInfo=trialFilesFrom['utah']['eventInfo'],
            spikeSourceType='nev',
            calcRigEvents=trialFilesFrom['utah']['calcRigEvents'],
            calcAverageLFP=True,
            removeMeanAcross=False,
            linearDetrend=False,
            interpolateOutliers=False, calcOutliers=False,
            normalizeByImpedance=False,
            asigNameList=asigNameList, ainpNameList=ainpNameList,
            nameSuffix='',
            writeMode='ow',
            chunkSize=chunkSize, equalChunks=equalChunks, chunkList=chunkList,
            LFPFilterOpts=None)
    ##
    if arguments['transferISIStimLog']:
        try:
            jsonSrcPath = os.path.join(nspFolder, ns5FileName + '_autoStimLog.json')
            jsonDestPath = trialBasePath.replace('.nix', '_autoStimLog.json')
            shutil.copyfile(jsonSrcPath, jsonDestPath)
        except Exception:
            traceback.print_exc()
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
    print('\n' + '#' * 50 + '\n{}\n{}\nComplete.\n'.format(dt.now().strftime('%Y-%m-%d %H:%M:%S'), __file__) + '#' * 50 + '\n')
