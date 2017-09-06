from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brpylib             import NevFile, brpylib_ver
import sys
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nevFile', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev')
parser.add_argument('--simiFile', default = 'Trial01_Step_Timing.txt')
args = parser.parse_args()
argNevFile = args.nevFile
argSimiFile = args.simiFile

# Plotting options
font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

simiName = '/' + argSimiFile.split('.')[0] + '_saveSimi.p'
simiFile = localDir + simiName
simiData = pd.read_pickle(simiFile)
simiDf = simiData['simiGait']
gaitLabelFun = simiData['gaitLabelFun']
upLabelFun = simiData['upLabelFun']
downLabelFun = simiData['downLabelFun']

spikeName = '/' + argNevFile.split('.')[0] + '_saveSpike.p'
spikeFile = localDir + spikeName
spikeData = pd.read_pickle(spikeFile)

spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}

tempUpLabels = assignLabels(binCenters, 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(binCenters, 'Toe Down', downLabelFun)
binnedLabels = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])

spikeMat['Labels'] = binnedLabels
spikeMat['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in spikeMat['Labels']])

with open(localDir + '/' + argNevFile.split('.')[0] + '_saveSpikeLabeled.p', "wb" ) as f:
    pickle.dump(spikeData, f, protocol=4 )

labelStruct = {'Labels': binnedLabels, 'LabelsNumeric' : pd.Series([labelsNumeric[x] for x in binnedLabels])}
with open(localDir + '/' + argNevFile.split('.')[0] + '_saveSpikeLabelsOnly.p', "wb" ) as f:
    pickle.dump(labelStruct, f, protocol=4 )

plotting = False
if plotting:
    #Plot the spikes
    chans = spikes['ChannelID']
    fi = plotBinnedSpikes(spikeMat.drop(['Labels', 'LabelsNumeric'], axis = 1), binCenters, chans, show = False)

    upMaskSpikes = (spikeMat['Labels'] == 'Toe Up').values
    downMaskSpikes = (spikeMat['Labels'] == 'Toe Down').values
    dummyVar = np.ones(binCenters.shape[0]) * 1
    ax = fi.axes[0]
    ax.plot(binCenters[upMaskSpikes], dummyVar[upMaskSpikes], 'ro')
    ax.plot(binCenters[downMaskSpikes], dummyVar[downMaskSpikes] + 1, 'go')
    plt.show(block = False)
