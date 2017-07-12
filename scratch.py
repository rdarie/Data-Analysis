argModel = 'bestSpikeLDA.pickle'
argModel = 'bestSpectrumLDA_DownSampled.pickle'
argFile = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev']
txtf = open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+ suffix + '.txt', 'w')
argFile ='201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'
import pandas as pd
from helper_functions import *
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
fileName = 'Log_Murdoc_01_06_2017_16_45_37.txt'

filePath = fileDir + '/' + fileName

log = readPiLog(filePath, names = ['Label', 'Time', 'Details'], zeroTime = True)
summary = pd.DataFrame(log['Label'].value_counts())
summary.index
eventDf = log
collapse = True
import plotly
plotly.tools.set_credentials_file(username='radudarie', api_key='g0dHEmrc2BEGVf6lHewl')
plot_events_raster(eventDf, names, usePlotly = True)
names = ['event', 'good', 'trial_start']
idx = 0
name = 'event'
len(eventDf['Time'][eventDf['Label'] == name])
plot_events_raster(eventDf, names, collapse = True, usePlotly = True)
fig.append_figure()

fig.append(tableFig, 1, 1)
plt.show()
len(data)
usePlotly = True
layout.keys()
len(layout['shapes'][0])
layout['shapes'][0]
fig.layout.xaxis6
bla = 1
dir(py)
import os
scriptPath = 'evaluatePerformance.py'
folderPath = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
import subprocess
runScriptAllFiles(scriptPath, folderPath)
onlyfiles
fileName = onlyfiles[1]
f = fileName
fileName = 'Log_Murdoc_08_06_2017_16_51_05.txt'
