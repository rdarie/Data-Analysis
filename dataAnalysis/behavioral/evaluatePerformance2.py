# Started November 2018; does not use plotly

import dataAnalysis.helperFunctions.helper_functions as hf
import proprioBehavioralControl.helperFunctions as bhf
import dataAnalysis.helperFunctions.motor_encoder as mea
import argparse, pdb
import os
import re
import seaborn as sns
import numpy as np
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file', default = 'Murdoc_29_09_2017_10_48_48',  nargs='*')
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')
parser.add_argument('--outputFileName')
args = parser.parse_args()

fileNamesRaw = args.file
if isinstance(fileNamesRaw, str):
    fileNamesRaw = [fileNamesRaw]
#fileNamesRaw = ['Log_Murdoc_2018_11_20_16_31_46.txt', 'Log_Murdoc_2018_11_20_16_46_43.txt', 'Log_Murdoc_2018_11_20_17_02_54.txt']

fileDir = args.folder
#fileDir = 'Y:\ENG_Neuromotion_Shared\group\Proprioprosthetics\Training\Flywheel Logs\Murdoc'

fileNames = []
for fileName in fileNamesRaw:
    if '.txt' in fileName:
        fileName = fileName.split('.txt')[0]
    if 'Log_' in fileName:
        fileName = fileName.split('Log_')[1]
    fileNames = fileNames + [fileName]

if args.outputFileName is not None:
    outputFileName = args.outputFileName
else:
    outputFileName = fileNames[0]
# outputFileName = 'Murdoc_2018_11_20'

filePaths = [fileDir + '/' + 'Log_' + fileName + '.txt' for fileName in fileNames]

conditionLongName = {
    'easy' : 'Cued by LED',
    'hard' : 'Uncued by LED',
    }
outcomeLongName = {
    'correct button' : 'Correct button',
    'incorrect button' : 'Incorrect button',
    'button timed out' : 'No press'
    }
typeLongName = {
    0 : 'Short First',
    1 : 'Long First'
    }

log, trialStats = hf.readPiJsonLog(filePaths, zeroTime = False)

# In[ ]: Count total # of events
with PdfPages(os.path.join(fileDir, outputFileName + '_piReport.pdf')) as pdf:
    eventList = log.loc[log['Label'] == 'event', 'Details']
    notLEDIndexes = eventList.index[np.flatnonzero(np.logical_not(eventList.str.contains('LED')))]
    ax = sns.countplot(x=eventList.loc[notLEDIndexes])
    ax.set_title('Total count of button presses and reward deliveries')

    pdf.savefig()
    plt.close()

    # In[ ]: Count choices
    ax = sns.countplot(x='Choice', data = trialStats)
    ax.set_title('Count of cued button presses')

    pdf.savefig()
    plt.close()

    # In[ ]:

    plotNames = ['goEasy', 'goHard', 'correct button', 'incorrect button']
    fi = hf.plot_events_raster(log, plotNames, collapse = True, usePlotly = False)
    plt.title('Events Raster')
    pdf.savefig()
    plt.close()
    # In[ ]: EasyPSTH
    """
    plotNames = ['correct button', 'incorrect button']
    stimulus = ['goEasy']
    preInterval = 2
    postInterval = 5
    deltaInterval = 500e-3

    fi = hf.plotPeristimulusTimeHistogram(log, stimulus, plotNames,
        preInterval = preInterval, postInterval = postInterval,
        deltaInterval = deltaInterval, usePlotly = False)

    pdf.savefig()
    plt.close()
    """
    # In[ ]: HardPSTH

    plotNames = ['correct button', 'incorrect button']
    stimulus = ['goHard']
    preInterval = 2
    postInterval = 5
    deltaInterval = 500e-3

    fi = hf.plotPeristimulusTimeHistogram(log, stimulus, plotNames,
        preInterval = preInterval, postInterval = postInterval,
        deltaInterval = deltaInterval, usePlotly = False)
    plt.title('Timing of button presses (PSTH)')
    pdf.savefig()
    pdf.savefig()
    plt.close()
    # In[ ]: Plot Trial Statistics
    trialStats['Condition'].replace(conditionLongName, inplace = True)
    trialStats['Outcome'].replace(outcomeLongName, inplace = True)
    trialStats['Type'].replace(typeLongName, inplace = True)

    nBins = 7
    stimIDs, stimIDsAbs, firstStimID, secondStimID = mea.getStimID(trialStats, nBins = nBins)

    trialStats["Stimulus ID Pair"]= stimIDs
    trialStats["Stimulus ID Pair (Abs)"]= stimIDsAbs

    trialStats["firstStimID"]= firstStimID.astype(float)
    trialStats["firstStimID (Abs)"]= firstStimID.astype(float).abs()
    trialStats["secondStimID"]= secondStimID.astype(float)
    trialStats["secondStimID (Abs)"]= secondStimID.astype(float).abs()

    trialStats.to_csv(fileDir + '/' + outputFileName + '_trialStats.csv')

    uncuedTrialStats = trialStats.loc[trialStats['Condition'] == 'Uncued by LED', :]
    ax = sns.countplot(x="secondStimID (Abs)", hue = 'Outcome', data=uncuedTrialStats)
    ax.set_title('Performance broken down by difficulty (Uncued)')

    pdf.savefig()
    plt.close()

    uncuedTrialStats['Outcome Score'] = uncuedTrialStats['Outcome']
    scoringDict = {
        'Correct button' : 100,
        'Incorrect button' :0,
        'No press':np.nan
        }
    uncuedTrialStats['Outcome Score'].replace(scoringDict, inplace = True)

    ax = sns.barplot(x="secondStimID (Abs)", y = 'Outcome Score', data=uncuedTrialStats)
    ax.set_title('Attempted trials broken down by difficulty (Uncued)')

    pdf.savefig()
    plt.close()
