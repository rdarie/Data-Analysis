
# coding: utf-8

# In[1]:

from helper_functions import *
import sys
sys.path.append('/home/pi/research/ProprioprostheticsBehavioralControl/')
from ProprioprostheticsBehavioralControl.helperFunctions import *
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.dashboard_objs as dashboard
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'Murdoc_26_06_2017_15_32_04')
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')
args = parser.parse_args()
fileName = args.file
fileDir = args.folder

if '.txt' in fileName:
    fileName = fileName.split('.txt')[0]
if 'Log_' in fileName:
    fileName = fileName.split('Log_')[1]

# In[2]:

filePath = fileDir + '/' + 'Log_' + fileName + '.txt'
log = readPiLog(filePath, names = ['Label', 'Time', 'Details'], zeroTime = True)

# In[3]:

counts = pd.DataFrame(log['Label'].value_counts())
table = ff.create_table(counts, index = True)
tableUrl = py.plot(table, filename= fileName + '/buttonPressSummary',fileopt="overwrite", sharing='public', auto_open=False)
#py.iplot(table, filename= fileName + '/buttonPressSummary',fileopt="overwrite", sharing='public')

# In[ ]:

data = [go.Bar(
            x=list(counts.index),
            y=counts.values
    )]

barUrl = py.plot(data, filename= fileName + '/buttonPressBar',fileopt="overwrite", sharing='public', auto_open=False)
#py.iplot(data, filename= fileName + '/buttonPressBar',fileopt="overwrite", sharing='public')

# In[ ]:

plotNames = ['trial_start', 'event', 'post_trial', 'good']
fi = plot_events_raster(log, plotNames, collapse = True, usePlotly = True)

rasterUrl = py.plot(fi, filename=fileName + '/buttonPressRaster',fileopt="overwrite", sharing='public', auto_open=False)
#py.iplot(fi, filename=fileName + '/buttonPressRaster',fileopt="overwrite", sharing='public')

# In[ ]:

existingDBoards = py.dashboard_ops.get_dashboard_names()

if fileName + '_dashboard' in existingDBoards:
    # If already exists, plots were updated above, just get the name
    dboard = py.dashboard_ops.get_dashboard(fileName + '_dashboard')
else:
    # If not, create the dashboard
    my_dboard = dashboard.Dashboard()

    fileIdBar = fileId_from_url(barUrl)
    fileIdRaster = fileId_from_url(rasterUrl)

    box_a = {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdBar,
        'title': 'BarPlot'
    }

    box_b = {
        'type': 'box',
        'boxType': 'plot',
        'fileId': fileIdRaster,
        'title': 'RasterPlot'
    }

    my_dboard.insert(box_a)
    my_dboard.insert(box_b, 'above', 1)

    dboardURL = py.dashboard_ops.upload(my_dboard, filename = fileName + '_dashboard', auto_open = False)
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
