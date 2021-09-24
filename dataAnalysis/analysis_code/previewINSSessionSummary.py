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
from copy import copy, deepcopy
from datetime import datetime, date
sns.set(
    context='talk', style='darkgrid',
    palette='dark', font='sans-serif',
    font_scale=0.75, color_codes=True)
import dataAnalysis.preproc.mdt as mdt
import dataAnalysis.preproc.mdt_constants as mdt_constants
import dataAnalysis.helperFunctions.helper_functions_new as hf
import rcsanalysis.packet_func as rcsa_helpers
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
        'hasStim': False,
        'tEnd': None,
        'duration': None,
        'maxAmp': None,
        'minAmp': None,
        'stimBreakdown': None
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
                    .value_counts())
                .to_frame(name='amplitudesTested'))
            summaryText += '<h3>Unique amplitudes: </h3>\n'
            summaryText += ampsPresent.to_html()
            logEntry['minAmp'] = int(ampsPresent['amplitudesTested'].min())
            logEntry['maxAmp'] = int(ampsPresent['amplitudesTested'].max())
            logEntry['hasStim'] = True
    except Exception:
        traceback.print_exc()
        summaryText += '<h3>Unable to read Stim Log</h3>\n'
    if len(HUTimestamps):
        hutDF = pd.concat(HUTimestamps, ignore_index=True)
        firstPacketT = pd.Timestamp(hutDF.min(), unit='ms', tz='EST')
        lastPacketT = pd.Timestamp(hutDF.max(), unit='ms', tz='EST')
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
        electrodeConfigurationCopy = deepcopy(electrodeConfiguration)
        summaryText += '<h2>Sense Configuration</h2>\n'
        summaryText += senseInfoDF.to_html()
        if logEntry['hasStim']:
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
            elecConfigDF.loc[:, 'cathodeStr'] = elecConfigDF.apply(
                lambda x: ''.join(['-E{:02d}'.format(ct) for ct in x['cathodes']]),
                axis='columns')
            elecConfigDF.loc[:, 'anodeStr'] = elecConfigDF.apply(
                lambda x: ''.join(['+E{:02d}'.format(ct) for ct in x['anodes']]),
                axis='columns')
            summaryText += '<h2>Stim Configuration</h2>\n'
            summaryText += elecConfigDF.loc[elecConfigDF['cathodes'].apply(lambda x: len(x) > 0), :].to_html()
        else:
            elecConfigDF = None
    except Exception:
        traceback.print_exc()
        elecConfigDF, senseInfoDF = None, None
        summaryText += '<h2>Device settings not read</h2>\n'
    if elecConfigDF is not None:
        try:
            progAmpNames = rcsa_helpers.progAmpNames
            expandCols = (
                    ['RateInHz', 'therapyStatus', 'trialSegment'] +
                    progAmpNames)
            deriveCols = ['amplitudeRound']
            stimStatusSerial = mdt.getINSStimLogFromJson(
                os.path.join(
                    orcaFolderPath, subjectName),
                ['Session{}'.format(sessionUnixTime)],
                deviceName=deviceName)
            stimStatusSerial.loc[:, 'HostUnixTime'] = stimStatusSerial['HostUnixTime'] / 1e3
            stimStatus = mdt.stimStatusSerialtoLong(
                stimStatusSerial, idxT='HostUnixTime', expandCols=expandCols,
                deriveCols=deriveCols, progAmpNames=progAmpNames,
                dummyTherapySwitches=False, elecConfiguration=electrodeConfigurationCopy
                )
            stimStatus = stimStatus.loc[stimStatus['amplitude'] != 0, :]
            stimStatus.rename(columns={'activeGroup': 'group'}, inplace=True)
            groupProgToElec = elecConfigDF.set_index(['group', 'program']).loc[stimStatus.set_index(['group', 'program']).index, :]
            elecCombo = groupProgToElec.apply(lambda x: '{}{}'.format(x['cathodeStr'], x['anodeStr']), axis='columns')
            stimStatus.loc[:, 'elecCombo'] = elecCombo.to_numpy()
            stimStatus.loc[:, 'cyclingEnabled'] = groupProgToElec['cyclingEnabled'].to_numpy()
            breakDownData, breakDownText, breakDownHtml = hf.calcBreakDown(
                    stimStatus,
                    breakDownBy=['elecCombo', 'cyclingEnabled', 'RateInHz', 'amplitude'])
            logEntry['stimBreakdown'] = breakDownData.reset_index().to_dict(orient='records')
            summaryText += '<h2>Stimulation pattern summary: </h2>\n'
            summaryText += breakDownHtml
        except Exception:
            traceback.print_exc()
            print('Did not print stim update summary')
            logEntry['stimBreakdown'] = None
            pdb.set_trace()
            summaryText += '<h2>Updates to stimulation not read</h2>\n'
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
    ####################################################################################################################
    cssTableStyles = [
        {
            'selector': 'th',
            'props': [
                ('border-style', 'solid'),
                ('border-color', 'black')]},
        {
            'selector': 'td',
            'props': [
                ('border-style', 'solid'),
                ('border-color', 'black')]},
        {
            'selector': 'table',
            'props': [
                ('border-collapse', 'collapse')
            ]}
        ]
    cm = sns.dark_palette("green", as_cmap=True)
    #
    summaryDF = pd.read_json(
        listOfSummarizedPath,
        orient='records',
        convert_dates=['tStart', 'tEnd'],
        dtype={
            'unixStartTime': int,
            'tStart': pd.DatetimeIndex,
            'tEnd': pd.DatetimeIndex,
            'hasTD': bool,
            'hasStim': bool,
            'duration': float,
            'maxAmp': int,
            'minAmp': int,
        })
    summaryDF.loc[:, 'tStart'] = pd.DatetimeIndex(summaryDF['tStart']).tz_convert("America/New_York")
    summaryDF.loc[:, 'tEnd'] = pd.DatetimeIndex(summaryDF['tEnd']).tz_convert("America/New_York")
    summaryDF.loc[:, 'experimentDate'] = summaryDF['tStart'].apply(lambda x: x.strftime('%Y-%m-%d'))
    summaryText = ''
    stimBoundsDict = {}
    for name, group in summaryDF.groupby('experimentDate'):
        summaryText += '<h2>{}</h2>\n'.format(name)
        if name in ['2018-12-13', '2018-12-14', '2018-12-17', '2018-12-18', '2018-12-19']:
            summaryText += '<h2>Skipped</h2>\n'
            continue
        stimSummaryDict = {}
        for rowIdx, row in group.iterrows():
            if row['stimBreakdown'] is not None:
                stimSummaryDict[row['unixStartTime']] = pd.DataFrame(row['stimBreakdown'])
        if (len(stimSummaryDict)):
            stimSummaryAll = pd.concat(stimSummaryDict, names=['unixStartTime', 'tempIdx']).reset_index().drop(columns=['tempIdx'])
            stimSummaryDF = stimSummaryAll.groupby(['elecCombo', 'cyclingEnabled', 'RateInHz', 'amplitude']).sum().loc[:, ['count']]
            stimMinDF = stimSummaryAll.groupby(['elecCombo', 'cyclingEnabled']).min().loc[:, ['amplitude']].rename(columns={'amplitude': 'min. amp.'})
            stimMaxDF = stimSummaryAll.groupby(['elecCombo', 'cyclingEnabled']).max().loc[:, ['amplitude']].rename(columns={'amplitude': 'max. amp.'})
            stimCountDF = stimSummaryAll.groupby(['elecCombo', 'cyclingEnabled']).sum().loc[:, ['count']]
            stimBoundsDict[name] = pd.concat([stimMinDF, stimMaxDF, stimCountDF], axis='columns')
            dfStyler = (
                stimSummaryDF
                .style
                .background_gradient(cmap=cm)
                .set_precision(1)
                )
            dfStyler.set_table_styles(cssTableStyles)
            summaryText += dfStyler.render()
            summaryText += '\n'
        else:
            summaryText += '<h3>No stim updates found.</h3>\n'
    stimSummaryPath = os.path.join(
        orcaFolderPath,
        subjectName + '_daily_stimulation_patterns.html')
    with open(stimSummaryPath, 'w') as _file:
        _file.write(summaryText)
    ####
    stimBoundsDF = pd.concat(stimBoundsDict, names=['date', 'elecCombo', 'cyclingEnabled']).reset_index()
    stimBoundsDF = stimBoundsDF.sort_values('elecCombo', kind='mergesort').set_index(['elecCombo', 'date', 'cyclingEnabled'])
    dfStyler = (
        stimBoundsDF
        .style
        .background_gradient(cmap=cm)
        .set_precision(1)
        .set_table_styles(cssTableStyles)
        )
    stimSummaryPath = os.path.join(
        orcaFolderPath,
        subjectName + '_electrode_stimulation_patterns.html')
    with open(stimSummaryPath, 'w') as _file:
        _file.write(dfStyler.render())
    ####################################################################################################################
    return


if __name__ == "__main__":
    summarizeINSSessionWrapper()
