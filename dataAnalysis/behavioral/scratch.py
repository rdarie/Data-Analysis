
fileName = 'Murdoc_31_08_2017_11_08_36'
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'

# for PSTH function
eventDf = log
names = ['red', 'blue', 'good']
stimulus = 'trial_start'

preInterval = 5
postInterval = 50
deltaInterval = 100e-3
nBins = int((preInterval + postInterval)/deltaInterval + 1)

for stimTime in log[log['Label'] == stimulus]['Time']:
    print(stimTime)

stimTime = log[log['Label'] == stimulus]['Time'][572]
name = 'red'
for idx, name in enumerate(names):
    pass

from dataAnalysis.helperFunctions.helper_functions import *

scriptPath = 'C:/Users/Radu/Documents/GitHub/Data-Analysis/dataAnalysis/behavioral/evaluatePerformance.py'
folderPath = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
runScriptAllFiles(scriptPath, folderPath)
