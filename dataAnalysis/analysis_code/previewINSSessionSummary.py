"""06b: Preprocess the INS Recording

Usage:
    viewINSSessionSummary.py [options]

Options:
    --sessionName=sessionName          which session to view
    --blockIdx=blockIdx                which trial to analyze [default: 1]
    --exp=exp                          which experimental day to analyze
    --reprocessAll                     Start analysis over [default: False]
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive qt output
# matplotlib.use('PS')   # generate offline postscript
from currentExperiment import parseAnalysisOptions
import seaborn as sns
import pandas as pd
import quantities as pq
import pdb, json, glob
from datetime import datetime, date
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=0.75, color_codes=True)
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.mdt_constants as mdt_constants
import os
from docopt import docopt
arguments = {arg.lstrip('-'): value for arg, value in docopt(__doc__).items()}
expOpts, allOpts = parseAnalysisOptions(
    int(arguments['blockIdx']), arguments['exp'])
globals().update(expOpts)
globals().update(allOpts)

import line_profiler
import atexit
import traceback
import json
from tqdm import tqdm

def summarizeINSSession(
        sessionUnixTime=None,
        subjectName=None,
        deviceName=None,
        orcaFolderPath=None
        ):
    summaryText = '<h1>Session{}</h1>\n'.format(sessionUnixTime)
    print(summaryText)
    #
    sessionTime = pd.Timestamp(sessionUnixTime, unit='ms', tz='EST')
    logEntry = {
        'unixStartTime': sessionUnixTime,
        'tStart': sessionTime.isoformat(),
        'hasTD': False,
        'tEnd': None,
        'duration': None,
        'maxAmp': None,
        'minAmp': None
        }
    summaryText += '<h2>Started: ' + sessionTime.strftime('%Y-%m-%d %H:%M:%S') + '</h2>\n'
    #
    HUTimestamps = []
    timeSyncPath = os.path.join(
        orcaFolderPath, subjectName,
        'Session{}'.format(sessionUnixTime),
        deviceName, 'TimeSync.json')
    if os.path.exists(timeSyncPath):
        try:
            timeSyncDF = mdt.getINSTimeSyncFromJson(
                os.path.join(
                orcaFolderPath, subjectName),
                ['Session{}'.format(sessionUnixTime)],
                deviceName=deviceName
                )
            HUTimestamps.append(timeSyncDF['HostUnixTime'])
            logEntry['hasTD'] = True
        except Exception:
            traceback.print_exc()
            summaryText += '<h3>TimeSync.json exists but not read</h3>\n'
    else:
        summaryText += '<h3>TimeSync.json does not exist</h3>\n'
    try:
        stimStatusSerial = mdt.getINSStimLogFromJson(
            os.path.join(
            orcaFolderPath, subjectName),
            ['Session{}'.format(sessionUnixTime)],
            deviceName=deviceName)
        HUTimestamps.append(stimStatusSerial['HostUnixTime'])
        therapyOnMask = ((stimStatusSerial['ins_property'] == 'therapyStatus') & (stimStatusSerial['ins_value'] == 1))
        if not therapyOnMask.any():
            summaryText += '<h3>Therapy never turned on!</h3>\n'
        ampOnMask = ((stimStatusSerial['ins_property'] == 'amplitude') & (stimStatusSerial['ins_value'] > 0))
        if not ampOnMask.any():
            summaryText += '<h3>No nonzero amplitude updates found!</h3>\n'
        else:
            ampsPresent = (
                pd.Series(
                    stimStatusSerial.loc[ampOnMask, 'ins_value']
                    .unique())
                .to_frame(name='amplitudesTested'))
            summaryText += '<h3>Unique amplitudes: </h3>\n'
            summaryText += ampsPresent.to_html()
            logEntry['minAmp'] = int(ampsPresent['amplitudesTested'].min())
            logEntry['maxAmp'] = int(ampsPresent['amplitudesTested'].max())
    except Exception:
        traceback.print_exc()
        summaryText += '<h3>Unable to read Stim Log</h3>\n'
    if len(HUTimestamps):
        hutDF = pd.concat(HUTimestamps, ignore_index=True)
        firstPacketT = pd.Timestamp(hutDF.min(), unit='ms')
        lastPacketT = pd.Timestamp(hutDF.max(), unit='ms')
        sessionDuration = lastPacketT - firstPacketT
        #
        logEntry['tEnd'] = lastPacketT.isoformat()
        logEntry['duration'] = float(sessionDuration.total_seconds())
        summaryText += '<h3>Duration: {} sec</h3>\n'.format(logEntry['duration'])
        summaryText += '<h3>Ended: {}</h3>\n'.format((sessionTime + sessionDuration).strftime('%Y-%m-%d %H:%M:%S'))
    else:
        summaryText += '<h3>Duration could not be determined. </h3>\n'
        summaryText += '<h3>End time could not be determined. </h3>\n'
    #
    try:
        electrodeConfiguration, senseInfoDF = mdt.getINSDeviceConfig(
            os.path.join(
                orcaFolderPath, subjectName),
            'Session{}'.format(sessionUnixTime),
            deviceName=deviceName)
        summaryText += '<h2>Sense Configuration</h2>\n'
        summaryText += senseInfoDF.to_html()
        elecConfigList = []
        for grpIdx, grpConfig in enumerate(electrodeConfiguration):
            for progIdx, progConfig in enumerate(grpConfig):
                thisCycleOnTime = (
                    progConfig['cycleOnTime']['time'] *
                    mdt_constants.cycleUnits[progConfig['cycleOnTime']['units']] *
                    pq.s).magnitude
                thisCycleOffTime = (
                    progConfig['cycleOffTime']['time'] *
                    mdt_constants.cycleUnits[progConfig['cycleOffTime']['units']] *
                    pq.s).magnitude
                progConfig['cycleOnTime'] = thisCycleOnTime
                progConfig['cycleOffTime'] = thisCycleOffTime
                progConfig['group'] = grpIdx
                progConfig['program'] = progIdx
                elecConfigList.append(pd.Series(progConfig))
        elecConfigDF = pd.concat(elecConfigList, axis='columns').transpose()
        summaryText += '<h2>Stim Configuration</h2>\n'
        summaryText += elecConfigDF.to_html()
    except Exception:
        traceback.print_exc()
        elecConfigDF, senseInfoDF = None, None
        summaryText += '<h2>Device settings not read</h2>\n'
    commentsJsonPath = os.path.join(
        orcaFolderPath, subjectName,
        'Session{}'.format(sessionUnixTime),
        '.MDT_SummitTest', 'Session{}'.format(sessionUnixTime) + '.json')
    if os.path.exists(commentsJsonPath):
        with open(commentsJsonPath, 'r') as f:
            commentsLog = json.load(f)
        summaryText += '<h2>Comments</h2>\n'
        summaryText += json.dumps(commentsLog)
    else:
        commentsLog = None
        summaryText += '<h2>Comments not found</h2>\n'
    print('#################################\n\n\n')
    return elecConfigDF, senseInfoDF, commentsLog, summaryText, logEntry


def summarizeINSSessionWrapper():
    # subjectName = 'Rupert'
    # deviceName = 'DeviceNPC700246H'
    # orcaFolderPath = '/gpfs/data/dborton/rdarie/Neural Recordings/ORCA Logs'
    orcaFolderPath = os.path.join(remoteBasePath, 'ORCA Logs')
    sessionFolders = sorted(
        glob.glob(os.path.join(orcaFolderPath, subjectName, 'Session*')))
    sessionUnixTimeList = [
        int(sessionName.split('Session')[-1])
        for sessionName in sessionFolders
        ]
    summaryPath = os.path.join(
        orcaFolderPath,
        subjectName + '_summary.html')
    listOfSummarizedPath = os.path.join(
        orcaFolderPath,
        subjectName + '_list_of_summarized.json'
        )
    ###
    if arguments['reprocessAll'] and os.path.exists(listOfSummarizedPath):
        os.remove(listOfSummarizedPath)
    if arguments['reprocessAll'] and os.path.exists(summaryPath):
        os.remove(summaryPath)
    ###
    if not os.path.exists(listOfSummarizedPath):
        listOfSummarized = []
        summaryDF = pd.DataFrame([], columns=['unixStartTime'])
        # with open(listOfSummarizedPath, 'w') as f:
        #     json.dump(listOfSummarized, f)
    else:
        with open(listOfSummarizedPath, 'r') as f:
            listOfSummarized = json.load(f)
            summaryDF = pd.DataFrame(listOfSummarized)
    for sessionUnixTime in tqdm(sessionUnixTimeList):
        if sessionUnixTime not in summaryDF['unixStartTime'].to_list():
            elecConfigDF, senseInfoDF, commentsLog, summaryText, logEntry = summarizeINSSession(
                sessionUnixTime=sessionUnixTime,
                subjectName=subjectName,
                deviceName=deviceName,
                orcaFolderPath=orcaFolderPath
                )
            with open(summaryPath, 'a+') as _file:
                _file.write(summaryText)
                _file.write('\n<hr>\n')
            listOfSummarized.append(logEntry)
    with open(listOfSummarizedPath, 'w') as f:
        json.dump(listOfSummarized, f)
    return


if __name__ == "__main__":
    summarizeINSSessionWrapper()
