argModel = 'bestSpikeLDA.pickle'
argModel = 'featureSelected_bestSpectrumLDA_DownSampled.pickle'
argFile = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev']
txtf = open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+ suffix + '.txt', 'w')
argFile ='201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'
import pandas as pd
from helper_functions import *
<<<<<<< HEAD:scratch.py
fileDir = 'C:/Users/radud/Desktop/Session1501886431276 - Release - Chan 1, 10, 14, 26/DeviceNPC700192H/'
fileName = 'Log_Murdoc_01_06_2017_16_45_37.txt'
=======
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
fileDir = 'Z:\data/rdarie/tempdata/Data-Analysis/Session1501885921457 - Debug - Chan 1, 10, 14, 26/DeviceNPC700192H'

estimator.steps
estimator.named_steps['downSampler'].kw_args['whichChans']
estimator.named_steps['featureSelector'].kw_args['support']
np.where(estimator.named_steps['featureSelector'].kw_args['support'])

192/96
>>>>>>> 005248b6dcfb3f55778827ee10b3461f2988fde0:dataAnalysis/lfpDecoding/scratch.py

filePath = fileDir + '/' + fileName
data = pd.read_table(filePath, sep='\t')
data.shape


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
import subprocess
onlyfiles
fileName = onlyfiles[1]
f = fileName
fileName = 'Log_Murdoc_08_06_2017_16_51_05.txt'


fileName = 'RawDataTD.json'
timeName = 'TimeSync.json'

data = pd.read_json(fileDir + '/' + fileName)
len(data['TimeDomainData'][0])
len(data['TimeDomainData'][0])