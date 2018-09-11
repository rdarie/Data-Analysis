# coding: utf-8

# In[1]:

from dataAnalysis.helperFunctions.helper_functions import *
from proprioBehavioralControl.helperFunctions import *
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.dashboard_objs as dashboard
import argparse, pdb
import copy
import re

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'Murdoc_29_09_2017_10_48_48',  nargs='*')
#fileNamesRaw = ['Log_Murdoc_2018_04_06_15_01_25.txt']
#fileNamesRaw = ['debugLog.txt']
"""
fileNamesRaw = ['Murdoc_2018_05_04_14_30_12',
    'Murdoc_2018_05_07_15_10_06',
    'Murdoc_2018_05_08_15_27_07',
    'Murdoc_2018_05_10_16_10_33',
    'Murdoc_2018_05_11_15_22_48'
]
"""
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')
#fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
#fileDir = 'C:/Users/Radu/Desktop'
parser.add_argument('--fixMovedToError', dest='fixMovedToError', action='store_true')
parser.add_argument('--outputFileName')
# outputFileName = 'Murdoc_2018_05_step_12_week_2'
parser.set_defaults(fixMovedToError = False)
args = parser.parse_args()

fileNamesRaw = args.file
if isinstance(fileNamesRaw, str):
    fileNamesRaw = [fileNamesRaw]

fileDir = args.folder
if args.outputFileName is not None:
    outputFileName = args.outputFileName
else:
    outputFileName = fileNamesRaw[0]

print(outputFileName)
fixMovedToError =args.fixMovedToError
#fixMovedToError = True
fileNames = []
for fileName in fileNamesRaw:
    if '.txt' in fileName:
        fileName = fileName.split('.txt')[0]
    if 'Log_' in fileName:
        fileName = fileName.split('Log_')[1]
    fileNames = fileNames + [fileName]

# In[2]:

#pdb.set_trace()
filePaths = [fileDir + '/' + 'Log_' + fileName + '.txt' for fileName in fileNames]
#filePaths = [fileDir + '/' + fileName + '.txt' for fileName in fileNames]

#log, trialStats = readPiLog(filePaths, names = ['Label', 'Time', 'Details'], zeroTime = True, fixMovedToError = [fixMovedToError for i in filePaths])
log, trialStats = readPiJsonLog(filePaths, zeroTime = True)

# In[3]:

countLog = copy.deepcopy(log)

for idx, row in countLog.iterrows():
    if row['Label'] == 'event':
        if (row['Details'] == 'green' or row['Details'] == 'red'):
            countLog.loc[idx, 'Label'] = 'Uncued ' + row['Details']
        if (row['Details'] == 'greenLED' or row['Details'] == 'redLED'):
            countLog.loc[idx, 'Label'] = row['Details']

counts = pd.DataFrame(countLog['Label'].value_counts())
table = ff.create_table(counts, index = True)
tableUrl = py.plot(table, filename= outputFileName + '/buttonPressSummary',fileopt="overwrite", sharing='public', auto_open=False)
#py.iplot(table, filename= fileName + '/buttonPressSummary',fileopt="overwrite", sharing='public')

# In[ ]:

data = [go.Bar(
    x=list(counts.index),
    y=counts.values
    )]

barUrl = py.plot(data, filename= outputFileName + '/buttonPressBar',fileopt="overwrite", sharing='public', auto_open=False)
#py.iplot(data, filename= fileName + '/buttonPressBar',fileopt="overwrite", sharing='public')

choiceCounts = trialStats['Choice'].value_counts()
data = [go.Bar(
    x=choiceCounts.index,
    y=choiceCounts.values
    )]
layout = go.Layout(
    title='Choices'
    )
choiceBarUrl = py.plot(data, filename= outputFileName + '/choiceBar',fileopt="overwrite", sharing='public', auto_open=False)


# In[ ]:

plotNames = ['goEasy', 'goHard', 'correct button', 'incorrect button']
fi = plot_events_raster(countLog, plotNames, collapse = True, usePlotly = True)

rasterUrl = py.plot(fi, filename= outputFileName + '/buttonPressRaster',
    fileopt="overwrite", sharing='public', auto_open=False)

# In[ ]: EasyPSTH

plotNames = ['correct button', 'incorrect button']
stimulus = ['goEasy']
preInterval = 2
postInterval = 5
deltaInterval = 500e-3

fi = plotPeristimulusTimeHistogram(countLog, stimulus, plotNames,
    preInterval = preInterval, postInterval = postInterval,
    deltaInterval = deltaInterval, usePlotly = True)

psthUrlEasy = py.plot(fi, filename= outputFileName + '/buttonPressPSTHEasy',
    fileopt="overwrite", sharing='public', auto_open=False)

# In[ ]: HardPSTH

plotNames = ['correct button', 'incorrect button']
stimulus = ['goHard']
preInterval = 2
postInterval = 5
deltaInterval = 500e-3

fi = plotPeristimulusTimeHistogram(countLog, stimulus, plotNames,
    preInterval = preInterval, postInterval = postInterval,
    deltaInterval = deltaInterval, usePlotly = True)

psthUrlHard = py.plot(fi, filename= outputFileName + '/buttonPressPSTHHard',
    fileopt="overwrite", sharing='public', auto_open=False)

# In[ ]:
outcomeLongName = {
    'correct button' : 'Correct button',
    'incorrect button' : 'Incorrect button',
    'button timed out' : 'No press'
    }
outcomeShortNames = sorted(outcomeLongName.keys())
outcomeLongNames = sorted(outcomeLongName.values())

fi, _, _ = plot_trial_stats(trialStats)
outcomeStatsUrl = py.plot(fi, filename= outputFileName + '/overallPercentages',
    fileopt="overwrite", sharing='public', auto_open=False)

fi, _, _ = plot_trial_stats(trialStats, separate = 'leftRight')
outcomeStatsByResponseUrl = py.plot(fi, filename= outputFileName + '/percentagesByResponse',
    fileopt="overwrite", sharing='public', auto_open=False)

fi, _, _ = plot_trial_stats(trialStats, separate = 'forwardBack')
outcomeStatsByDirUrl = py.plot(fi, filename= outputFileName + '/percentagesByDir',
    fileopt="overwrite", sharing='public', auto_open=False)

trialStats.to_csv(fileDir + '/' + outputFileName + '_trialStats.csv')

############ trial durations
totalMagnitude = np.abs(trialStats.loc[:, 'First'].values) + np.abs(trialStats.loc[:, 'Second'].values)
totalMagnitude = np.asarray([float(item) for item in totalMagnitude])
#totalMagnitude.dtype

durations = trialStats.loc[:, 'Stimulus Duration'].values
durations = np.asarray([float(item) for item in durations])

# Create a trace
trace = go.Scatter(
    x = durations,
    y = totalMagnitude,
    mode = 'markers'
    )

data = [trace]
stimDurationUrl = py.plot(data, filename= outputFileName + '/stimDuration',
    fileopt="overwrite", sharing='public', auto_open=False)


# agregate by stimulus duration
shortestStimDur = trialStats['Stimulus Duration'].min()
longestStimDur = trialStats['Stimulus Duration'].max()
bins = np.linspace(shortestStimDur, longestStimDur, 10)
binAssignments = np.digitize([float(value) for value in trialStats['Stimulus Duration']],bins)

binNames = {i : '%2.1f to %2.1f' %(bins[i-1], bins[i]) for i in range(1,len(bins))}
binNames.update({0: '< %2.1f' % bins[0]})
binNames.update({len(bins): '> %2.1f' % bins[-1]})

trialStatsGrouped = trialStats.groupby(binAssignments)

stimBinnedStats = pd.DataFrame(0, index = outcomeLongNames,
    columns = sorted(binNames.keys()))
plotXAxisEntries = [value[1] for value in binNames.items()]
#name, group = next(iter(trialStatsGrouped))
for name, group in trialStatsGrouped:
    #idx, row = next(group.iterrows())
    _, data, layout = plot_trial_stats(group)
    for datum in data:
        #datum = next(iter(data))
        findNums = re.search(r'\d+', datum['name']).span()
        numTrialStart = findNums[0]
        numTrialStop = findNums[1]
        condition = datum['name'][:numTrialStart - 1]

        if condition == 'Uncued by LED':
            print(condition)
            print(datum['x'])
            print(datum['y'])

            for idx, outcome in enumerate(datum['x']):
                #idx, outcome = next(enumerate(datum['x']))
                stimBinnedStats.loc[outcome, name] = datum['y'][idx]

stimBinnedPlotData = []
for name in outcomeLongNames:
    stimBinnedPlotData.append(go.Bar(
        x= plotXAxisEntries,
        y= stimBinnedStats.loc[name, :],
        name=name
        ))

layout['title'] = 'Outcomes by Stimulus Duration'
fig = go.Figure(data=stimBinnedPlotData, layout=layout)
stimBinnedPlotUrl = py.plot(fig, filename= outputFileName + '/percentagesByStimDur',fileopt="overwrite", sharing='public', auto_open=False)

# Aggregate by stimulus pair
def getStimID(trialStats):
    #trialStats.columns
    allMagnitudes = pd.concat([trialStats.loc[:,'First'].abs(), trialStats.loc[:,'Second'].abs()], ignore_index = True)
    binEdges = np.linspace(0.99 * allMagnitudes.min(), 1.01 * allMagnitudes.max(), 10)
    firstStimID  = pd.cut(trialStats.loc[:,'First'].abs(), binEdges, labels = [i for i in range(9)])
    secondStimID = pd.cut(trialStats.loc[:,'Second'].abs(), binEdges, labels = [i for i in range(9)])
    stimIDs = pd.Series([(firstStimID[i], secondStimID[i]) for i in trialStats.index])
    return stimIDs

if "Stimulus ID Pair" not in trialStats.columns:
    stimIDs = getStimID(trialStats)
    trialStats.insert(0, "Stimulus ID Pair", stimIDs)

trialStatsByStimID = trialStats.groupby(by="Stimulus ID Pair")
type(trialStatsByStimID.size())


plotXAxisEntries = ['(%d, %d), %d total' % (name[0], name[1], count) for name, count in trialStatsByStimID.size().items()]

pairBinnedStats = pd.DataFrame(0, index = outcomeLongNames,
    columns = sorted(trialStatsByStimID.groups.keys()))
#name, group = next(iter(trialStatsByStimID))
for name, group in trialStatsByStimID:
    #idx, row = next(group.iterrows())
    _, data, layout = plot_trial_stats(group)
    for datum in data:
        #datum = next(iter(data))
        findNums = re.search(r'\d+', datum['name']).span()
        numTrialStart = findNums[0]
        numTrialStop = findNums[1]
        condition = datum['name'][:numTrialStart - 1]

        if condition == 'Uncued by LED':
            print(condition)
            print(datum['x'])
            print(datum['y'])

            for idx, outcome in enumerate(datum['x']):
                #idx, outcome = next(enumerate(datum['x']))
                pairBinnedStats.loc[outcome, name] = datum['y'][idx]

#pdb.set_trace()
pairBinnedPlotData = []
for name in outcomeLongNames:
    pairBinnedPlotData.append(go.Bar(
        x= plotXAxisEntries,
        y= pairBinnedStats.loc[name, :],
        name=name
        ))

layout['title'] = 'Outcomes by Stimulus Pair'
fig = go.Figure(data=pairBinnedPlotData, layout=layout)
stimBinnedPlotUrl = py.plot(fig, filename= outputFileName + '/percentagesByStimPair',fileopt="overwrite", sharing='public', auto_open=False)

################################################################################
# Plot statistics about the blocks
trialStats = trialStats.reset_index(drop = True)
#idx, row = next(trialStats.iterrows())

blockStats = pd.DataFrame(columns = ['First Outcome', 'First Condition', 'Type', 'Length'])
blockIdx = 0
for idx, row in trialStats.iterrows():
    if idx == 0:
        blockStats.loc[0, 'Type'] = row['Type']
        blockStats.loc[0, 'First Condition'] = row['Condition']
        blockStats.loc[0, 'First Outcome'] = row['Outcome']
        blockStats.loc[0, 'Length'] = 1
    else:
        if trialStats.loc[idx - 1, 'Type'] == row['Type']:
            #if the previous trial was of the same type as the current trial:
            blockStats.loc[blockIdx, 'Length'] = blockStats.loc[blockIdx, 'Length'] + 1
        else:
            #start a new block
            blockIdx = blockIdx + 1
            blockStats.loc[blockIdx, 'Type'] = row['Type']
            blockStats.loc[blockIdx, 'First Outcome'] = row['Outcome']
            blockStats.loc[blockIdx, 'First Condition'] = row['Condition']
            blockStats.loc[blockIdx, 'Length'] = 1

blockStats.to_csv(fileDir + '/' + outputFileName + '_blockStats.csv')

newNames = {
    'Type' : 'Type',
    'First Outcome' : 'Outcome',
    'First Condition' : 'Condition',
    'Length' : 'Length'
    }

fi, _, _ = plot_trial_stats(blockStats.rename(columns = newNames), title = 'Outcomes for first trial of each block')
blockOutcomeStatsUrl = py.plot(fi, filename= outputFileName + '/FirstInBlockPercentages',
    fileopt="overwrite", sharing='public', auto_open=False)

blockLengths = blockStats['Length'].value_counts()
data = [go.Bar(
            x=list(blockLengths.index),
            y=blockLengths.values
    )]

blockBarUrl = py.plot(data, filename= outputFileName + '/blockBar',fileopt="overwrite", sharing='public', auto_open=False)

# plot block stats by block trials

extendedColumns = ['Result of trial %s' % str(s + 1) for s in range(max(blockStats['Length']))]
extendedBlockStats = pd.DataFrame(0, index = outcomeShortNames,
    columns = extendedColumns)

trialIdx = 1
totals = {name : 0 for name in extendedBlockStats}
#idx, row = next(trialStats.iterrows())
for idx, row in trialStats.iterrows():
    if idx == 0:
        if row['Condition'] == 'hard':
            extendedBlockStats.loc[row['Outcome'], 'Result of trial 1'] = 1
    else:
        trialIdx = trialIdx + 1 if trialStats.loc[idx - 1, 'Type'] == row['Type'] else 1
        if row['Condition'] == 'hard':
            extendedBlockStats.loc[row['Outcome'], 'Result of trial %s' % str(trialIdx)] += 1

#name = next(iter(extendedBlockStats))
for name in extendedBlockStats:
    totals[name] = extendedBlockStats[name].sum()
    extendedBlockStats[name] = extendedBlockStats[name] / extendedBlockStats[name].sum()

plotXAxisEntries = [' '.join([name, ', ', str(totals[name])]) + ' total trials' for name in extendedBlockStats]
extendedBlockStats =extendedBlockStats.rename(index = outcomeLongName)

#name = next(iter(outcomeShortNames))
data = []
for name in outcomeShortNames:
    data.append(go.Bar(
        x= plotXAxisEntries,
        y= extendedBlockStats.loc[outcomeLongName[name], :],
        name=outcomeLongName[name]
        ))

layout = {
    'barmode' : 'group',
    'title' : 'Block Outcomes, Trial by Trial',
    'xaxis' : {
        'title' : 'Outcome',
        'titlefont' : {
            'family' : 'Arial',
            'size' : 30,
            'color' : '#7f7f7f'
            },
        'tickfont' : {
            'size' : 20
            }
        },
    'yaxis' : {
        'title' : 'Percentage',
        'titlefont' : {
            'family' : 'Arial',
            'size' : 25,
            'color' : '#7f7f7f'
            },
        'tickfont' : {
            'size' : 20
            }
        },
    'legend' : {
        'font' : {
            'size' : 20
        }
    }
    }

fig = go.Figure(data=data, layout=layout)
extendedBlockOutcomesUrl = py.plot(fig, filename= outputFileName + '/percentagesWithinBlock',fileopt="overwrite", sharing='public', auto_open=False)


""" """
# In[ ]:

existingDBoards = py.dashboard_ops.get_dashboard_names()

if outputFileName + '_dashboard' in existingDBoards:
    # If already exists, plots were updated above, just get the name
    dboard = py.dashboard_ops.get_dashboard(outputFileName + '_dashboard')
else:
    # If not, create the dashboard
    my_dboard = dashboard.Dashboard()
    fileIdBar = fileId_from_url(barUrl)
    fileIdRaster = fileId_from_url(rasterUrl)
    fileIdPsthEasy = fileId_from_url(psthUrlEasy)
    fileIdPsthHard = fileId_from_url(psthUrlHard)
    fileIdOutcomes = fileId_from_url(outcomeStatsUrl)
    fileIdOutcomesByDir = fileId_from_url(outcomeStatsByDirUrl)
    fileIdOutcomesByResponse = fileId_from_url(outcomeStatsByResponseUrl)
    fileIdBlockOutcomes = fileId_from_url(blockOutcomeStatsUrl)
    fileIdBlockBar = fileId_from_url(blockBarUrl)
    fileIDExtendedBlockOutcomes = fileId_from_url(extendedBlockOutcomesUrl)

    boxes = [{
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdBar,
        'title': 'BarPlot'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdRaster,
        'title': 'RasterPlot'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdPsthEasy,
        'title': 'PSTH Plot Easy'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdPsthHard,
        'title': 'PSTH Plot Hard'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdOutcomes,
        'title': 'Overall Percentages'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdBlockOutcomes,
        'title': 'First Trial in Every Block Percentages'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdBlockBar,
        'title': 'BlockBarPlot'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdOutcomesByDir,
        'title': 'Overall Percentages by Direction'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdOutcomesByResponse,
        'title': 'Overall Percentages by Response'
    },

    {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIDExtendedBlockOutcomes,
        'title': 'Percentages in each Block'
    }]


    my_dboard.insert(boxes[0])
    my_dboard.insert(boxes[1], 'above', 1)
    my_dboard.insert(boxes[2], 'above', 1)
    my_dboard.insert(boxes[3], 'above', 1)
    my_dboard.insert(boxes[4], 'above', 1)
    my_dboard.insert(boxes[5], 'above', 1)
    my_dboard.insert(boxes[6], 'above', 1)
    my_dboard.insert(boxes[7], 'above', 1)
    my_dboard.insert(boxes[8], 'above', 1)
    my_dboard.insert(boxes[9], 'above', 1)

    my_dboard['layout']['size'] = 1000

    dboardURL = py.dashboard_ops.upload(my_dboard, filename = outputFileName + '_dashboard', auto_open = False)

    #dboardID = fileId_from_url(dboardURL)

# In[ ]:

if 'dboardURL' in locals():
   credentials = get_gsheets_credentials()
   http = credentials.authorize(httplib2.Http())
   discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                   'version=v4')
   service = discovery.build('sheets', 'v4', http=http,
                             discoveryServiceUrl=discoveryUrl)

   spreadsheetId = '1BWjBqbtoVr9j6dU_7eHp-bQMJApNn8Wkl_N1jv20faE'
   range_name = 'Sheet1!T1'
   valueInputOption = 'USER_ENTERED'

   body = {
     'values': [['=MATCH(\"'+ fileName + '\",H:H,0)']]
   }

   result = service.spreadsheets().values().update(
       spreadsheetId = spreadsheetId, range=range_name,
       valueInputOption=valueInputOption, body=body).execute()

   matchingIdx = service.spreadsheets().values().get(
       spreadsheetId=spreadsheetId, range=range_name).execute()

   put_range_name = 'Sheet1!D' + matchingIdx['values'][0][0]

   body = {
     'values': [[dboardURL]]
   }

   result = service.spreadsheets().values().update(
       spreadsheetId = spreadsheetId, range=put_range_name,
       valueInputOption=valueInputOption, body=body).execute()
