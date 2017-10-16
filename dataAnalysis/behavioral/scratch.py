
fileNamesRaw = ['Log_Murdoc_13_10_2017_17_20_22.txt']
fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
outputFileName = 'Murdoc_13_10_2017_17_20_22'
fixMovedToError = False
# for PSTH function
eventDf = log

names = plotNames
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

len(data)

scriptPath = 'C:/Users/Radu/Documents/GitHub/Data-Analysis/dataAnalysis/behavioral/evaluatePerformance.py'
folderPath = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
outputFileName = 'Murdoc_29_09_03_10'
runScriptAllFiles(scriptPath, folderPath)
usePlotly = True
fileNamesRaw = ['Murdoc_29_09_2017_10_48_48', 'Murdoc_03_10_2017_10_54_10']
it = enumerate(names)
idx, name = next(it)

"""
firstThrow = None
secondThrow = None

firstThrow = copy.deepcopy(log[log['Label'].str.contains('turnPedalRandom_1')].loc[:, ['Details']].reset_index())
secondThrow = copy.deepcopy(log[log['Label'].str.contains('turnPedalRandom_2')].loc[:, ['Details']].reset_index())
trialStats = pd.concat([firstThrow['Details'], secondThrow['Details']], axis = 1)
trialStats.columns = ['First', 'Second']

difference = firstThrow.subtract(secondThrow)['Details']

trialStats['Difficulty'] = pd.Series([
    magnitude_lookup_table(thisDifference) for thisDifference in difference
    ])

mask = np.logical_or(log['Label'].str.contains('easy') ,
    log['Label'].str.contains('easy'))
log[mask].size
"""
