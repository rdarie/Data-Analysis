from dataAnalysis.helperFunctions.helper_functions import *
import traceback
try:
    from proprioBehavioralControl.helperFunctions import *
except Exception:
    traceback.print_exc()
try:
    import plotly.plotly as py
    import plotly.tools as tls
    import plotly.figure_factory as ff
    import plotly.graph_objs as go
    import plotly.dashboard_objs as dashboard
except Exception:
    traceback.print_exc()
import argparse, pdb
import copy
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cbook as cbook

import seaborn as sns
sns.set(font_scale=1.75)

#fileName = 'Murdoc_2018_01_31_16_07_13'
def name_to_date(fileName, spec):
    parts = [int(name) for name in fileName.split('_')[1:]]
    if spec == 'old':
        dateName = datetime.datetime(parts[2], parts[1], parts[0], parts[3], parts[4], parts[5])
    else:
        dateName = datetime.datetime(parts[0], parts[1], parts[2], parts[3], parts[4], parts[5])
    return dateName


# morning trials
"""
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
"""
fileList = [
    [
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
'Murdoc_18_01_2018_10_48_13',
'Murdoc_19_01_2018_10_53_20',
'Murdoc_22_01_2018_10_42_48',
'Murdoc_22_01_2018_11_06_37',
'Murdoc_23_01_2018_10_45_57',
'Murdoc_23_01_2018_11_17_00',
'Murdoc_24_01_2018_10_56_58',
'Murdoc_24_01_2018_11_25_40',
'Murdoc_25_01_2018_10_50_48',
'Murdoc_25_01_2018_11_27_12',
'Murdoc_26_01_2018_11_01_54',
'Murdoc_29_01_2018_10_44_34',
'Murdoc_29_01_2018_11_17_16',
'Murdoc_30_01_2018_10_39_15',
'Murdoc_30_01_2018_11_15_44',
'Murdoc_30_01_2018_14_59_41',
    ],[
'Murdoc_2018_01_31_10_33_11',
'Murdoc_2018_01_31_16_07_13',
'Murdoc_2018_02_01_10_38_28',
'Murdoc_2018_02_01_15_18_16',
'Murdoc_2018_02_02_10_43_56',
'Murdoc_2018_02_04_12_09_03',
'Murdoc_2018_02_05_10_45_40',
'Murdoc_2018_02_06_10_58_16',
'Murdoc_2018_02_06_14_38_55',
'Murdoc_2018_02_07_10_21_54',
'Murdoc_2018_02_08_10_46_06',
'Murdoc_2018_02_08_16_21_26',
'Murdoc_2018_02_09_10_57_31',
'Murdoc_2018_02_11_11_19_49',
'Murdoc_2018_02_13_10_44_52',
'Murdoc_2018_02_14_10_46_25',
'Murdoc_2018_02_15_10_49_48',
'Murdoc_2018_02_16_10_32_28',
'Murdoc_2018_02_16_11_28_41'
    ]
    ]
"""
# afternoon trials
fileList = [
    'Murdoc_30_01_2018_14_59_41',
    'Murdoc_2018_01_31_16_07_13',
    'Murdoc_2018_02_01_15_18_16',
    'Murdoc_2018_02_04_12_09_03',
    'Murdoc_2018_02_06_14_38_55',
    'Murdoc_2018_02_08_16_21_26',
    ]
    """
sessionTimes = [name_to_date(name, 'old') for name in fileList[0]] + [name_to_date(name, 'new') for name in fileList[1]]
#fileName = fileList[0]
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
choiceLongName = {
    'green' : 'A > B button',
    'red' : 'B > A button',
    np.nan : 'No Press'
    }

overallTrialStats = pd.DataFrame(index = sessionTimes, columns = pd.MultiIndex.from_product([['Cued by LED', 'Uncued by LED'], ['Correct button', 'Incorrect button', 'No press']]))
overallBlockStats = pd.DataFrame(index = sessionTimes, columns = pd.MultiIndex.from_product([['Cued by LED', 'Uncued by LED'], ['Correct button', 'Incorrect button', 'No press']]))

overallTrialChoices = pd.DataFrame(index = sessionTimes, columns = pd.MultiIndex.from_product([['Cued by LED', 'Uncued by LED'], ['A > B button', 'B > A button', 'No Press']]))

for idx, fileName in enumerate(fileList[0] + fileList[1]):
    try:
        trialStats = pd.read_csv(fileDir + '/' + fileName + '_trialStats.csv')
    except:
        trialStats = pd.read_csv(fileDir + '/' + 'Log_' + fileName + '_trialStats.csv')
    print('Running %s' % fileName)
    for conditionName in np.unique(trialStats['Condition']):
        #condition is easy vs hard:
        conditionStats = trialStats[trialStats['Condition'] == conditionName]

        #y counts the number of outcomes
        y = [conditionStats[conditionStats['Outcome'] == on].size \
            for on in (np.unique(conditionStats['Outcome']))]
        #x is the name of the outcomes
        x = [outcomeLongName[on] for on in (np.unique(conditionStats['Outcome']))]

        for subIdx, val in enumerate(y):
            overallTrialStats.loc[sessionTimes[idx], (conditionLongName[conditionName], x[subIdx])] = 100 * val / sum(y)

        #y counts the number of choices
        y = [conditionStats[conditionStats['Choice'] == on].size \
            for on in (conditionStats['Choice'].unique())]
        #x is the name of the choice
        x = [choiceLongName[on] for on in (conditionStats['Choice'].unique())]

        for subIdx, val in enumerate(y):
            overallTrialChoices.loc[sessionTimes[idx], (conditionLongName[conditionName], x[subIdx])] = 100 * val / sum(y)
    try:
        blockStats = pd.read_csv(fileDir + '/' + fileName + '_blockStats.csv')
    except:
        blockStats = pd.read_csv(fileDir + '/' + 'Log_' + fileName + '_blockStats.csv')

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
            overallBlockStats.loc[sessionTimes[idx], (conditionLongName[conditionName], x[subIdx])] = 100 * val / sum(y)

plt.subplot(2,1,1)
for column in overallBlockStats.loc[:, 'Cued by LED']:
    plt.plot(overallBlockStats.loc[:, ('Uncued by LED', column)], 'o-', label = column)

plt.legend(loc = 1)
plt.title('Trials at the beginning of each block')
plt.ylabel('Percentage')

plt.subplot(2,1,2)
for column in overallTrialStats.loc[:, 'Uncued by LED']:
    plt.plot(overallTrialStats.loc[:, ('Uncued by LED', column)], 'o-', label = column)

#plt.legend()
plt.title('All trials')
plt.ylabel('Percentage')
plt.xlabel('Day')

fig = plt.gcf()
fig.suptitle("Trials uncued by LED - Outcomes", fontsize=22)
plt.show()

plt.subplot(2,1,1)
for column in overallBlockStats.loc[:, 'Cued by LED']:
    plt.plot(overallBlockStats.loc[:, ('Cued by LED', column)], 'o-', label = column)

plt.legend(loc = 1)
plt.title('Trials at the beginning of each block')
plt.ylabel('Percentage')

plt.subplot(2,1,2)
for column in overallTrialStats.loc[:, 'Cued by LED']:
    plt.plot(overallTrialStats.loc[:, ('Cued by LED', column)], 'o-', label = column)

plt.legend()
plt.title('All trials')
plt.ylabel('Percentage')

plt.xlabel('Day')
fig = plt.gcf()
fig.suptitle("Trials cued by LED - Outcomes", fontsize=22)
plt.show()

# ----------------- choice stats --------------------------------------------------------------
plt.subplot(1,1,1)

for column in overallTrialChoices.loc[:, 'Uncued by LED']:
    plt.plot(overallTrialChoices.loc[:, ('Uncued by LED', column)], 'o-', label = column)

#plt.legend()
plt.title('All trials')
plt.ylabel('Percentage')
plt.xlabel('Day')
plt.legend(loc = 1)

fig = plt.gcf()
fig.suptitle("Trials uncued by LED - Choices", fontsize=22)
plt.show()

plt.subplot(1,1,1)
for column in overallTrialChoices.loc[:, 'Cued by LED']:
    plt.plot(overallTrialChoices.loc[:, ('Cued by LED', column)], 'o-', label = column)

plt.legend()
plt.title('All trials')
plt.ylabel('Percentage')

plt.xlabel('Day')
plt.legend(loc = 1)
fig = plt.gcf()
fig.suptitle("Trials cued by LED - Choices", fontsize=22)
plt.show()
