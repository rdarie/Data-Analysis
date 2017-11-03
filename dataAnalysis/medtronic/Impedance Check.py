
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import cv2
import os
# In[1]:
# Plotting options

"""
Isaacson, Benjamin
Aug 28 (1 day ago)

to David, me, Marc, David, Evan
Lead impedance pulses used:
The INS will provide the capability to measure electrode impedance within the following ranges:
Amplitude=2 mA; Pulse width=80 us; Rate=100 Hz: >=50 and <=300 ohms
Amplitude=0.4 mA; Pulse width=80 us; Rate=100 Hz: >=300 and <=4,500 ohms
Amplitude=0.1 mA; Pulse width=330 us; Rate=100 Hz: >=4500 and <=40,000 ohms

"""

font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 100
        }
fig_opts = {
    'figsize' : (16,12),
    'dpi' : 100
    }

sns.set(font_scale=3)
sns.set_style("white")
matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)
baseFolder = 'Z:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Spinal Electrode/Microleads impedance testing'
insFilePath = [
    #Part 1
    [baseFolder + '/INS/trial_1.txt',
        baseFolder + '/INS/trial_3.txt'],
    #Part 2
    [baseFolder + '/INS/trial_2.txt',
        baseFolder + '/INS/trial_4.txt']
]

data = {'Impedance':np.array([]), 'Electrode':np.array([]), 'Method':[]}
info = {'Flags':[], 'Electrode':np.array([])}
originalOrder = list(range(32))
INSOrder = list(reversed(range(8))) + list(reversed(range(8, 16))) + list(reversed(range(16,24))) + list(reversed(range(24,32)))
electrodeIDCorrespondence = {i:j for i,j in zip(INSOrder, originalOrder)}
elecIDtoPW = {
0.0:330,
1.0:330,
2.0:80,
3.0:330,
4.0:330,
5.0:330,
6.0:330,
7.0:330,
8.0:80,
9.0:80,
10.0:80,
11.0:80,
12.0:80,
13.0:330,
14.0:330,
15.0:330,
16.0:330,
17.0:80,
18.0:80,
19.0:80,
20.0:330,
21.0:330,
22.0:330,
23.0:330,
24.0:80,
25.0:80,
26.0:80,
27.0:80,
28.0:80,
29.0:80,
30.0:80,
31.0:80
}

elecArea = {
0.0:0.13,
1.0:0.28,
2.0:0.42,
3.0:0.219,
4.0:0.161,
5.0:0.166,
6.0:0.189,
7.0:0.189,
8.0:0.189,
9.0:0.225,
10.0:0.254,
11.0:0.262,
12.0:0.274,
13.0:0.209,
14.0:0.149,
15.0:0.147,
16.0:0.18,
17.0:0.416,
18.0:0.176,
19.0:0.186
}
# Get impedance readings
for filePath in insFilePath[0]:
    with open(filePath) as f:
        for line in f:
            idx = line.find('Impedance [')
            if idx != -1:
                elecIdx = line.split('[')[1].split(']')[0]
                impVal = line.split('[')[1].split(':')[1]

                #Correct to kOhm
                data['Impedance'] = np.append(data['Impedance'], float(impVal)*1e-3)
                data['Electrode'] = np.append(data['Electrode'], electrodeIDCorrespondence[int(elecIdx)])
                data['Method'].append('INS')

# In[5]:
for filePath in insFilePath[1]:
    with open(filePath) as f:
        for line in f:
            idx = line.find('Impedance [')
            if idx != -1:
                elecIdx = line.split('[')[1].split(']')[0]
                impVal = line.split('[')[1].split(':')[1]

                #Correct to kOhm
                data['Impedance'] = np.append(data['Impedance'], float(impVal)*1e-3)
                data['Electrode'] = np.append(data['Electrode'], electrodeIDCorrespondence[int(elecIdx) + 16])
                data['Method'].append('INS')
#

# Get other info
with open(insFilePath[0][1]) as f:
    for line in f:
        idx = line.find('Info [')
        if idx != -1:
            elecIdx = line.split('[')[1].split(']')[0]
            flags = line.split('[')[1].split(':')[1].split(',')

            info['Electrode'] = np.append(info['Electrode'], electrodeIDCorrespondence[int(elecIdx)])
            info['Flags'].append(flags)

# In[5]:

with open(insFilePath[1][1]) as f:
    for line in f:
        idx = line.find('Info [')
        if idx != -1:
            elecIdx = line.split('[')[1].split(']')[0]
            flags = line.split('[')[1].split(':')[1].split(',')

            info['Electrode'] = np.append(info['Electrode'], electrodeIDCorrespondence[int(elecIdx) + 16])
            info['Flags'].append(flags)


info = pd.DataFrame(info)
info = info.drop_duplicates(subset=['Electrode']).sort_values('Electrode')


nanozFilePath = baseFolder + 'Microleads_Impedance_test.xlsx'
nanozData = [pd.read_excel(nanozFilePath,sheetname=0), pd.read_excel(nanozFilePath,sheetname=1)]

nanozData[0].drop(nanozData[0].columns[[0,1,2,-1]],1,inplace = True)
nanozData[0] = nanozData[0].loc[range(20)] * 1e-3

nanozData[1] = nanozData[1].loc[range(20)]
nanozData[1].drop(nanozData[1].columns[[0,1,2,6,10]],1,inplace = True)

nanozData = pd.concat(nanozData, axis = 1)

for index, row in nanozData.iterrows():
    #Correct to kOhm
    data['Impedance'] = np.append(data['Impedance'], row.values * 1e3)
    data['Electrode'] = np.append(data['Electrode'], np.ones(row.values.shape) * index)
    data['Method'] = data['Method'] + ['nanoZ' for i in row.values]

data = pd.DataFrame(data)

import copy

INSData = copy.deepcopy(data)

for index, row in INSData.iterrows():
    if row['Method'] == 'nanoZ':
        INSData.loc[index, 'Impedance'] = 0

nanoZData = copy.deepcopy(data)

for index, row in nanoZData.iterrows():
    if row['Method'] == 'INS':
        nanoZData.loc[index, 'Impedance'] = 0

figNames = []
for i in np.unique(data['Electrode']):

    fig, ax = plt.subplots()
    ax2 =ax.twinx()

    sns.barplot(x = 'Electrode', y = 'Impedance', ax = ax2, hue = 'Method', data=nanoZData[nanoZData['Electrode'] == i])
    sns.barplot(x = 'Electrode', y = 'Impedance', ax = ax, hue = 'Method', data=INSData[INSData['Electrode'] == i])
    ax.set_ylabel('Impedance (kOhm) - '+str(elecIDtoPW[i])+' us square wave')
    ax2.set_ylabel('Impedance (kOhm) - 1kHz sine wave')

    ax.legend_.remove()
    ax2.legend_.remove()
    ax.legend(loc = 'upper right')

    figName = '_{:02d}.png'.format(int(i))
    figNames.append(figName)
    #plt.tight_layout()
    plt.savefig(baseFolder + figName)

    plt.close()

fig, ax = plt.subplots()
ax2 =ax.twinx()

sns.barplot(x = 'Electrode', y = 'Impedance', ax = ax2, hue = 'Method', data=nanoZData[nanoZData['Electrode'] < 20])
sns.barplot(x = 'Electrode', y = 'Impedance', ax = ax, hue = 'Method', data=INSData[INSData['Electrode'] < 20])
ax.set_ylabel('Impedance (kOhm) - ' + 'square wave')
ax2.set_ylabel('Impedance (kOhm) - 1kHz sine wave')

ax.legend_.remove()
ax2.legend_.remove()
ax.legend(loc = 'upper right')
plt.savefig(baseFolder + '_all.png')

scopePics = ['00_Microleads-BSI20_Zoom_01.tif',
 '01_Microleads-BSI20_Zoom_03.tif',
 '02_Microleads-BSI20_Zoom_04.tif',
 '03_Microleads-BSI20_Zoom_02.tif',
 '04_Microleads-BSI20_Zoom_08.tif',
 '05_Microleads-BSI20_Zoom_07.tif',
 '06_Microleads-BSI20_Zoom_05.tif',
 '07_Microleads-BSI20_Zoom_06.tif',
 '08_Microleads-BSI20_Zoom_09.tif',
 '09_Microleads-BSI20_Zoom_10.tif',
 '10_Microleads-BSI20_Zoom_11.tif',
 '11_Microleads-BSI20_Zoom_12.tif',
 '12_Microleads-BSI20_Zoom_20.tif',
 '13_Microleads-BSI20_Zoom_17.tif',
 '14_Microleads-BSI20_Zoom_18.tif',
 '15_Microleads-BSI20_Zoom_19.tif',
 '16_Microleads-BSI20_Zoom_16.tif',
 '17_Microleads-BSI20_Zoom_13.tif',
 '18_Microleads-BSI20_Zoom_15.tif',
 '19_Microleads-BSI20_Zoom_14.tif']

i = 0
blankImage = np.zeros((20 * 1200, 2 * 1600,3), np.uint8)

x0_p = 0
x1_p = 1600
x0_s = 1600
x1_s = 3200

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 2
thickness = 10
whiteTuple = (255,255,255)
textSize = cv2.getTextSize('250 um', font, fontScale, thickness)
import pdb
for i in range(20):
    singleTile = np.zeros((1200, 2 * 1600,3), np.uint8)
    y0 = i * 1200
    y1 = y0 + 1200
    thePlot = cv2.imread(baseFolder + figNames[i])

    blankImage[y0:y1, x0_p:x1_p] = thePlot
    singleTile[0:1200, x0_p:x1_p] = thePlot

    theScopePic = cv2.imread(baseFolder + scopePics[i])

    blankImage[y0:y1, x0_s:x1_s] = theScopePic
    singleTile[0:1200, x0_s:x1_s] = theScopePic

    blankImage[(y0 + 150):(y0 + 175), (x0_s + 200):(x0_s + 400)] = np.ones((25, 200,3), np.uint8)*255
    singleTile[150:175, (x0_s + 200):(x0_s + 400)] = np.ones((25, 200,3), np.uint8)*255

    # Write some Text
    singleTile = cv2.putText(singleTile,'250 um',(x0_s + 200, 300), font, fontScale, whiteTuple, thickness, cv2.LINE_AA)
    blankImage = cv2.putText(blankImage,'250 um',(x0_s + 200,y0 + 300), font, fontScale, whiteTuple, thickness, cv2.LINE_AA)
    #pdb.set_trace()
    cv2.imwrite(baseFolder + 'tiled_{:02d}.png'.format(int(i)), singleTile)

cv2.imwrite(baseFolder + 'tiled.png', blankImage)

info['Mean Impedance (INS)'] = pd.Series()
info['Mean Impedance (nanoZ)'] = pd.Series()
info['Area'] = pd.Series()
for index, row in info.iterrows():
    INSMask = np.logical_and(
        data['Electrode'].isin([info.loc[index, 'Electrode']]),
        data['Method'].isin(['INS'])
    )
    info.loc[index, ('Mean Impedance (INS)')] = np.mean(data['Impedance'][INSMask])

    nanoZMask = np.logical_and(
        data['Electrode'].isin([info.loc[index, 'Electrode']]),
        data['Method'].isin(['nanoZ'])
    )
    info.loc[index, ('Mean Impedance (nanoZ)')] = np.mean(data['Impedance'][nanoZMask])

    try:
        info.loc[index, ('Area')] = elecArea[info.loc[index, 'Electrode']]
    except:
        pass

fig, ax = plt.subplots()
ax2 = ax.twinx()

sns.set_style("dark")
sns.regplot(x = info['Area'],
    y = info['Mean Impedance (nanoZ)'],
    ax = ax, label = 'nanoZ', ci = None)
ax.set_xlabel('Electrode exposed area (mm^2)')
ax.set_ylabel('Impedance (kOhm) - 1kHz Sine')
sns.regplot(x = info['Area'],
    y = info['Mean Impedance (INS)'],
    ax = ax2, label = 'INS', ci = None)
ax2.set_ylabel('Impedance (kOhm) - square wave')

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
plt.legend([h1[0] , h2[0]], [l1[0] , l2[0]], loc = 'best')
ax.set_ylim([0,140]); ax2.set_ylim([0,16])
plt.savefig(baseFolder + 'size_effect')
