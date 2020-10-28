"""06b: Preprocess the INS Recording

Usage:
    viewINSSessionSummary.py [options]

Options:
    --sessionName=sessionName          which session to view
"""

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
matplotlib.use('Qt5Agg')   # generate interactive qt output
# matplotlib.use('PS')   # generate offline postscript
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

import line_profiler
import atexit
import traceback


def summarizeINSSession(
        sessionUnixTime=None,
        subjectName=None,
        deviceName=None,
        orcaFolderPath=None
        ):
    summaryText = '<h1>Session{}</h1>\n'.format(sessionUnixTime)
    #
    sessionTime = datetime.fromtimestamp(sessionUnixTime / 1e3)
    summaryText += '<h3>Started: ' + sessionTime.isoformat() + '</h3>\n'
    #
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
            firstPacketT = datetime.fromtimestamp(timeSyncDF['HostUnixTime'].min() / 1e3)
            lastPacketT = datetime.fromtimestamp(timeSyncDF['HostUnixTime'].max() / 1e3)
            summaryText += '<h3>Duration: {}</h3>\n'.format(lastPacketT - firstPacketT)
        except Exception:
            traceback.print_exc()
            summaryText += '<h3>Duration: TimeSync.json exists but not read</h3>\n'
    else:
        summaryText += '<h3>Duration: NA</h3>\n'
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
        '.MDT_SummitTest','Session{}'.format(sessionUnixTime) + '.json')
    if os.path.exists(commentsJsonPath):
        with open(commentsJsonPath, 'r') as f:
            commentsLog = json.load(f)
        summaryText += '<h2>Comments</h2>\n'
        summaryText += json.dumps(commentsLog)
    else:
        commentsLog = None
        summaryText += '<h2>Comments not found</h2>\n'
    return elecConfigDF, senseInfoDF, commentsLog, summaryText


def summarizeINSSessionWrapper():
    subjectName = 'Rupert'
    deviceName = 'DeviceNPC700246H'
    orcaFolderPath = '/gpfs/data/dborton/rdarie/Neural Recordings/ORCA Logs'
    # pdb.set_trace()
    sessionFolders = sorted(
        glob.glob(os.path.join(orcaFolderPath, subjectName, 'Session*')))
    sessionUnixTimeList = [
        int(sessionName.split('Session')[-1])
        for sessionName in sessionFolders
        ]
    summaryPath = os.path.join(
        orcaFolderPath, subjectName,
        'summary.html')
    if os.path.exists(summaryPath):
        os.remove(summaryPath)
    for sessionUnixTime in sessionUnixTimeList:
        elecConfigDF, senseInfoDF, commentsLog, summaryText = summarizeINSSession(
            sessionUnixTime=sessionUnixTime,
            subjectName=subjectName,
            deviceName=deviceName,
            orcaFolderPath=orcaFolderPath
            )
        with open(summaryPath, 'a+') as _file:
            _file.write(summaryText)
            _file.write('\n<hr>\n')
    return


if __name__ == "__main__":
    summarizeINSSessionWrapper()
