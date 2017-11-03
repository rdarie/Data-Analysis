
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

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'Murdoc_29_09_2017_10_48_48',  nargs='*')
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')
parser.add_argument('--fixMovedToError', dest='fixMovedToError', action='store_true')
parser.add_argument('--outputFileName', required = True)
parser.set_defaults(fixMovedToError = False)
args = parser.parse_args()

fileNamesRaw = args.file
if isinstance(fileNamesRaw, str):
    fileNamesRaw = [fileNamesRaw]

fileDir = args.folder
outputFileName = args.outputFileName
fixMovedToError =args.fixMovedToError

fileNames = []
for fileName in fileNamesRaw:
    if '.txt' in fileName:
        fileName = fileName.split('.txt')[0]
    if 'Log_' in fileName:
        fileName = fileName.split('Log_')[1]
    fileNames = fileNames + [fileName]

# In[2]:
filePaths = [fileDir + '/' + 'Log_' + fileName + '.txt' for fileName in fileNames]

log = readPiLog(filePaths, names = ['Label', 'Time', 'Details'], zeroTime = True, fixMovedToError = [True, True])

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

mask = log['Label'].str.contains('turnPedalRandom_1') | \
    log['Label'].str.contains('turnPedalRandom_2') | \
    log['Label'].str.contains('easy') | \
    log['Label'].str.contains('hard') | \
    log['Label'].str.endswith('correct button') | \
    log['Label'].str.endswith('incorrect button') | \
    log['Label'].str.endswith('button timed out')
trialRelevant = pd.DataFrame(log[mask]).reset_index()
# TODO: kludge, to avoid wait for corect button substring. fix

while not trialRelevant.iloc[-1, :]['Label'] in ['correct button', 'incorrect button', 'button timed out']:
    trialRelevant.drop(trialRelevant.index[len(trialRelevant)-1], inplace = True)

def magnitude_lookup_table(difference):
    if difference > 2e4:
        return 'Long Extension'
    if difference > 0:
        return 'Short Extension'
    if difference > -2e4:
        return 'Short Flexion'
    else:
        return 'Long Flexion'

trialStartIdx = trialRelevant.index[trialRelevant['Label'].str.contains('turnPedalRandom_1')]
trialStats = pd.DataFrame(index = trialStartIdx, columns = ['First', 'Second', 'Magnitude', 'Condition', 'Outcome'])
for idx in trialStartIdx:
    assert trialRelevant.loc[idx, 'Label'] == 'turnPedalRandom_1'
    trialStats.loc[idx, 'First'] = float(trialRelevant.loc[idx, 'Details'])
    assert trialRelevant.loc[idx + 1, 'Label'] == 'turnPedalRandom_2'
    trialStats.loc[idx, 'Second'] = float(trialRelevant.loc[idx + 1, 'Details'])
    trialStats.loc[idx, 'Magnitude'] = magnitude_lookup_table(trialStats.loc[idx, 'First'] - trialStats.loc[idx, 'Second'])
    assert (trialRelevant.loc[idx + 2, 'Label'] == 'easy') | (trialRelevant.loc[idx + 2, 'Label'] == 'hard')
    trialStats.loc[idx, 'Condition'] = trialRelevant.loc[idx + 2, 'Label']
    assert (trialRelevant.loc[idx + 3, 'Label'] == 'correct button') | \
        (trialRelevant.loc[idx + 3, 'Label'] == 'incorrect button') | \
        (trialRelevant.loc[idx + 3, 'Label'] == 'button timed out')
    trialStats.loc[idx, 'Outcome'] = trialRelevant.loc[idx + 3, 'Label']

fi = plot_trial_stats(trialStats)
outcomeStatsUrl = py.plot(fi, filename= outputFileName + '/buttonPressOutcomeStats',
    fileopt="overwrite", sharing='public', auto_open=False)


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
        'title': 'OutcomesPlot'
    }]


    my_dboard.insert(boxes[0])
    my_dboard.insert(boxes[1], 'above', 1)
    my_dboard.insert(boxes[2], 'above', 1)
    my_dboard.insert(boxes[3], 'above', 1)
    my_dboard.insert(boxes[4], 'above', 1)

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
   range_name = 'Sheet1!J1'
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
