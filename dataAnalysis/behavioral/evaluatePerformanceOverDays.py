from dataAnalysis.helperFunctions.helper_functions import *
from proprioBehavioralControl.helperFunctions import *
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.dashboard_objs as dashboard
import argparse, pdb
import copy
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

fileList = [
    'Murdoc_17_10_2017_11_00_41',
    'Murdoc_19_10_2017_10_28_51',
    'Murdoc_24_10_2017_10_55_56',
    'Murdoc_25_10_2017_10_55_50',
    'Murdoc_27_10_2017_11_24_22',
    'Murdoc_31_10_2017_10_54_54',
    'Murdoc_01_11_2017_10_59_26',
    'Murdoc_06_11_2017_10_49_52',
    'Murdoc_09_11_2017_10_47_21',
    'Murdoc_17_11_2017_11_14_22',
    'Murdoc_20_11_2017_10_49_56',
    'Murdoc_21_11_2017_10_49_04',
    'Murdoc_22_11_2017_10_37_55',
    'Murdoc_27_11_2017_11_04_42',
    'Murdoc_28_11_2017_11_04_28',
    'Murdoc_29_11_2017_10_18_34',
    'Murdoc_29_11_2017_10_42_18',
    'Murdoc_30_11_2017_11_37_27',
    'Murdoc_01_12_2017_10_44_06',
    'Murdoc_04_12_2017_10_57_13',
    'Murdoc_06_12_2017_10_58_28',
    'Murdoc_07_12_2017_10_37_06',
    'Murdoc_07_12_2017_11_05_49',
    'Murdoc_08_12_2017_10_44_25',
    'Murdoc_08_12_2017_10_52_01',
    'Murdoc_10_12_2017_11_41_22',
    'Murdoc_10_12_2017_12_01_22',
    'Murdoc_11_12_2017_10_49_49',
    'Murdoc_11_12_2017_11_02_25',
    'Murdoc_12_12_2017_10_55_54',
    'Murdoc_13_12_2017_10_59_52',
    'Murdoc_13_12_2017_11_20_24',
    'Murdoc_14_12_2017_10_34_20',
    'Murdoc_15_12_2017_11_26_51',
    'Murdoc_15_12_2017_11_39_00',
    'Murdoc_18_12_2017_10_48_36',
    'Murdoc_19_12_2017_10_46_03',
    'Murdoc_20_12_2017_10_33_22',
    'Murdoc_20_12_2017_11_21_29',
    'Murdoc_21_12_2017_10_51_20',
    'Murdoc_10_01_2018_11_35_56',
    'Murdoc_11_01_2018_10_51_33',
    'Murdoc_12_01_2018_10_43_35',
    'Murdoc_12_01_2018_11_07_48',
    'Murdoc_14_01_2018_12_10_56',
    'Murdoc_15_01_2018_11_44_47',
    'Murdoc_17_01_2018_11_16_09',
    ]

fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc'
#fileName = fileList[0]

conditionLongName = {
    'easy' : 'Cued by LED',
    'hard' : 'Uncued by LED'
    }
outcomeLongName = {
    'correct button' : 'Correct button',
    'incorrect button' : 'Incorrect button',
    'button timed out' : 'No press'
    }


overallTrialStats = pd.DataFrame(index = range(len(fileList)), columns = pd.MultiIndex.from_product([['Cued by LED', 'Uncued by LED'], ['Correct button', 'Incorrect button', 'No press']]))
overallBlockStats = pd.DataFrame(index = range(len(fileList)), columns = pd.MultiIndex.from_product([['Cued by LED', 'Uncued by LED'], ['Correct button', 'Incorrect button', 'No press']]))

for idx, fileName in enumerate(fileList):
    trialStats = pd.read_csv(fileDir + '/' + fileName + '_trialStats.csv')

    for conditionName in np.unique(trialStats['Condition']):
        #condition is easy vs hard:
        conditionStats = trialStats[trialStats['Condition'] == conditionName]
        y = [conditionStats[conditionStats['Outcome'] == on].size \
            for on in sorted(np.unique(conditionStats['Outcome']))]
        x = [outcomeLongName[on] for on in sorted(np.unique(conditionStats['Outcome']))]

        for subIdx, val in enumerate(y):
            overallTrialStats.loc[idx, (conditionLongName[conditionName], x[subIdx])] = 100 * val / sum(y)

    blockStats = pd.read_csv(fileDir + '/' + fileName + '_blockStats.csv')
    newNames = {
        'Type' : 'Type',
        'First Outcome' : 'Outcome',
        'First Condition' : 'Condition',
        'Length' : 'Length'
        }
    blockStats = blockStats.rename(columns = newNames)

    for conditionName in np.unique(blockStats['Condition']):
        #condition is easy vs hard:
        conditionStats = blockStats[blockStats['Condition'] == conditionName]
        y = [conditionStats[conditionStats['Outcome'] == on].size \
            for on in sorted(np.unique(conditionStats['Outcome']))]
        x = [outcomeLongName[on] for on in sorted(np.unique(conditionStats['Outcome']))]

        for subIdx, val in enumerate(y):
            overallBlockStats.loc[idx, (conditionLongName[conditionName], x[subIdx])] = 100 * val / sum(y)

plt.subplot(2,1,1)
for column in overallTrialStats.loc[:, 'Cued by LED']:
    plt.plot(overallTrialStats.loc[:, ('Cued by LED', column)], label = column)

plt.legend()
plt.title('Cued by LED')
plt.ylabel('Percentage Correct')

plt.subplot(2,1,2)
for column in overallTrialStats.loc[:, 'Uncued by LED']:
    plt.plot(overallTrialStats.loc[:, ('Uncued by LED', column)], label = column)

plt.legend()
plt.title('Uncued by LED')
plt.ylabel('Percentage Correct')
plt.xlabel('Day')
fig = plt.gcf()
fig.suptitle("Overall Trial Stats", fontsize=14)
plt.show()

plt.subplot(2,1,1)
for column in overallBlockStats.loc[:, 'Cued by LED']:
    plt.plot(overallBlockStats.loc[:, ('Cued by LED', column)], label = column)

plt.legend()
plt.title('Cued by LED')
plt.ylabel('Percentage Correct')

plt.subplot(2,1,2)
for column in overallBlockStats.loc[:, 'Cued by LED']:
    plt.plot(overallBlockStats.loc[:, ('Uncued by LED', column)], label = column)

plt.legend()
plt.title('Uncued by LED')
plt.ylabel('Percentage Correct')
plt.xlabel('Day')
fig = plt.gcf()
fig.suptitle("First in Block Stats", fontsize=14)
plt.show()
