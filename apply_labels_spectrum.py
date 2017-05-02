from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brpylib             import NevFile, brpylib_ver
import sys
import pickle
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

ns5Name = '/saveSpectrumRight.p'
ns5File = localDir + ns5Name
data = pd.read_pickle(ns5File)

simiName = '/saveSimi.p'
simiFile = localDir + simiName
simiData = pd.read_pickle(simiFile)
simiDf = simiData['simiGait']
gaitLabelFun = simiData['gaitLabelFun']
upLabelFun = simiData['upLabelFun']
downLabelFun = simiData['downLabelFun']

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}

tempUpLabels = assignLabels(data['channel']['t'], 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(data['channel']['t'], 'Toe Down', downLabelFun)
binnedLabels = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])
data['channel']['data']['Labels'] = binnedLabels
data['channel']['data']['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in data['channel']['data']['Labels']])

tempUpLabels = assignLabels(data['channel']['spectrum']['t'], 'Toe Up', upLabelFun)
tempDownLabels = assignLabels(data['channel']['spectrum']['t'], 'Toe Down', downLabelFun)
binnedLabels = pd.Series([x if x == 'Toe Up' else y for x,y in zip(tempUpLabels, tempDownLabels)])
data['channel']['spectrum']['Labels'] = binnedLabels
data['channel']['spectrum']['LabelsNumeric'] = pd.Series([labelsNumeric[x] for x in data['channel']['spectrum']['Labels']])

with open(localDir + "/saveSpectrumRightLabeled.p", "wb" ) as f:
    pickle.dump(data, f, protocol=4 )

labelStruct = {'Labels': binnedLabels, 'LabelsNumeric' : pd.Series([labelsNumeric[x] for x in binnedLabels])}
with open(localDir + "/saveSpectrumRightLabelsOnly.p", "wb" ) as f:
    pickle.dump(labelStruct, f, protocol=4 )

plotting = False
if plotting:
    #plot the spectrum
    upMaskSpectrum = (data['channel']['spectrum']['Labels'] == 'Toe Up').values
    downMaskSpectrum = (data['channel']['spectrum']['Labels'] == 'Toe Down').values
    dummyVar = np.ones(data['channel']['spectrum']['t'].shape[0]) * 1

    fi = plotSpectrum(data['channel']['spectrum']['PSD'][1],
        data['channel']['samp_per_s'],
        data['channel']['start_time_s'],
        data['channel']['t'][-1],
        fr = data['channel']['spectrum']['fr'],
        t = data['channel']['spectrum']['t'],
        show = False)
    ax = fi.axes[0]
    ax.plot(data['channel']['spectrum']['t'][upMaskSpectrum], dummyVar[upMaskSpectrum], 'ro')
    ax.plot(data['channel']['spectrum']['t'][downMaskSpectrum], dummyVar[downMaskSpectrum] + 1, 'go')
    #ax.plot(data['channel']['spectrum']['t'][stanceMask], dummyVar[stanceMask], 'go')
    plt.show(block = False)

    # plot one channel
    f,ax = plotChan(data['channel'], 25, mask = None, show = False)
    dummyVar = np.ones(data['channel']['t'].shape[0]) * 100
    upMaskChan = (data['channel']['data']['Labels'] == 'Toe Up').values
    downMaskChan = (data['channel']['data']['Labels'] == 'Toe Down').values
    ax.plot(data['channel']['t'][downMaskChan], dummyVar[downMaskChan], 'go')
    ax.plot(data['channel']['t'][upMaskChan], dummyVar[upMaskChan], 'ro')
    plt.show()
