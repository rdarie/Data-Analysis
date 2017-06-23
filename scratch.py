argModel = 'bestSpikeLDA.pickle'
argModel = 'bestSpectrumLDA_DownSampled.pickle'
argFile = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev']
txtf = open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+ suffix + '.txt', 'w')
argFile ='201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'
import pandas as pd
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
fileName = 'Log_Murdoc_08_06_2017_16_36_32.txt'

filePath = fileDir + '/' + fileName

log = readPiLog(filePath, names = ['Label', 'Time', 'Details'])
log['Label'].value_counts()

eventDf = log
