from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os
from sklearn.metrics import confusion_matrix

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

ns5Name = '/saveSpectrumRightLabeled.p'
ns5File = localDir + ns5Name
ns5Data = pd.read_pickle(ns5File)

whichChans = range(96)
whichFreqs = ns5Data['channel']['spectrum']['fr'] < 300

spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
labels = ns5Data['channel']['spectrum']['LabelsNumeric']
y = labels

modelName = '/bestSpectrumLogReg.pickle'
modelFile = localDir + modelName
estimator = pd.read_pickle(modelFile)['estimator']

# get all columns of spikemat that aren't the labels
chans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

X = spikeMat[chans]
y = spikeMat['LabelsNumeric']

yHat = estimator.predict(X)
ylogreg=bestLogReg.predict(X)

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
numericLabels = {v: k for k, v in labelsNumeric.items()}
predictedLabels = pd.Series([numericLabels[x] for x in ylogreg])

plotting = True
if plotting:
    #plot the spectrum
    upMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'] == 'Toe Up').values
    downMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'] == 'Toe Down').values
    dummyVar = np.ones(ns5Data['channel']['spectrum']['t'].shape[0]) * 1

    upMaskSpectrumPredicted = (predictedLabels == 'Toe Up').values
    downMaskSpectrumPredicted = (predictedLabels == 'Toe Down').values

    fi = plotSpectrum(ns5Data['channel']['spectrum']['PSD'][1],
        ns5Data['channel']['samp_per_s'],
        ns5Data['channel']['start_time_s'],
        ns5Data['channel']['t'][-1],
        fr = ns5Data['channel']['spectrum']['fr'],
        t = ns5Data['channel']['spectrum']['t'],
        show = False)
    ax = fi.axes[0]
    ax.plot(ns5Data['channel']['spectrum']['t'][upMaskSpectrum], dummyVar[upMaskSpectrum], 'ro')
    ax.plot(ns5Data['channel']['spectrum']['t'][downMaskSpectrum], dummyVar[downMaskSpectrum] + 1, 'go')

    ax.plot(ns5Data['channel']['spectrum']['t'][upMaskSpectrumPredicted], dummyVar[upMaskSpectrumPredicted] + .5, 'mo')
    ax.plot(ns5Data['channel']['spectrum']['t'][downMaskSpectrumPredicted], dummyVar[downMaskSpectrumPredicted] + 1.5, 'co')

    with open(localDir + '/spikePlot.pickle', 'wb') as f:
        pickle.dump(fi, f)
    with open(localDir + '/spikeConfusionMatrix.pickle', 'wb') as f:
        pickle.dump(fiCm, f)
