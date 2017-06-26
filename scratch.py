argModel = 'bestSpikeLDA.pickle'
argModel = 'bestSpectrumLDA_DownSampled.pickle'
argFile = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev']
txtf = open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+ suffix + '.txt', 'w')
argFile ='201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'
import pandas as pd
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
fileName = 'Log_Murdoc_01_06_2017_16_45_37.txt'

filePath = fileDir + '/' + fileName

log = readPiLog(filePath, names = ['Label', 'Time', 'Details'], zeroTime = True)
log['Label'].value_counts()

eventDf = log
import plotly
plotly.tools.set_credentials_file(username='radudarie', api_key='g0dHEmrc2BEGVf6lHewl')
plot_events_raster(eventDf, names)
idx = 0
names = ['event', 'good', 'trial_start']
name = 'event'
len(eventDf['Time'][eventDf['Label'] == name])
plot_events_raster(eventDf, names, collapse = False)
plt.show()
plotly.__version__
