from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os
from sklearn.metrics import confusion_matrix, f1_score

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

spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
fr = ns5Data['channel']['spectrum']['fr']
labels = ns5Data['channel']['spectrum']['LabelsNumeric']

nSamples = len(t)
y = labels

whichChans = [0, 24, 49, 95]
whichFreqs = ns5Data['channel']['spectrum']['fr'] < 100
flatSpectrum = spectrum[whichChans, :, whichFreqs].transpose(1, 0, 2).to_frame().transpose()
X = flatSpectrum

# Poor man's test train split:
trainSize = 0.9
trainIdx = slice(None, int(trainSize * nSamples))
testIdx = slice(int(trainSize * nSamples) + 1, None)

modelName = '/bestSpectrumLogReg.pickle'
modelFile = localDir + modelName
estimatorDict = pd.read_pickle(modelFile)
estimator = estimatorDict['estimator']
estimatorInfo = estimatorDict['info']

estimator.fit(X.iloc[trainIdx, :], y.iloc[trainIdx])
yHat = estimator.predict(X.iloc[testIdx, :])

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
numericLabels = {v: k for k, v in labelsNumeric.items()}
predictedLabels = pd.Series([numericLabels[x] for x in yHat])

# Compute confusion matrix
cnf_matrix = confusion_matrix(y.iloc[testIdx], yHat)
print("Normalized confusion matrix:")
cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

# Compute F1 score
f1Score = f1_score(y.iloc[testIdx], yHat, average = 'weighted')
print("F1 Score was:")
print(f1Score)

plotting = True
if plotting:
    #plot the spectrum
    upMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'].iloc[testIdx] == 'Toe Up').values
    downMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'].iloc[testIdx] == 'Toe Down').values
    dummyVar = np.ones(t[testIdx].shape[0]) * 1

    upMaskSpectrumPredicted = (predictedLabels == 'Toe Up').values
    downMaskSpectrumPredicted = (predictedLabels == 'Toe Down').values

    fi = plotSpectrum(spectrum[1].iloc[testIdx],
        ns5Data['channel']['samp_per_s'],
        t[testIdx][0],
        t[testIdx][-1],
        fr = fr,
        t = t[testIdx],
        show = False)
    ax = fi.axes[0]
    ax.plot(t[testIdx][upMaskSpectrum], dummyVar[upMaskSpectrum], 'ro')
    ax.plot(t[testIdx][downMaskSpectrum], dummyVar[downMaskSpectrum] + 1, 'go')

    ax.plot(t[testIdx][upMaskSpectrumPredicted], dummyVar[upMaskSpectrumPredicted] + .5, 'mo')
    ax.plot(t[testIdx][downMaskSpectrumPredicted], dummyVar[downMaskSpectrumPredicted] + 1.5, 'co')

    # Plot normalized confusion matrix
    fiCm = plotConfusionMatrix(cnf_matrix, classes = labelsNumeric.keys(), normalize=True,
                          title='Normalized confusion matrix')

    #plt.show()
    figDic = {'spectrum': fi, 'confusion': fiCm}

    with open(localDir + '/spikePlot.pickle', 'wb') as f:
        pickle.dump(fi, f)
    with open(localDir + '/spikeConfusionMatrix.pickle', 'wb') as f:
        pickle.dump(fiCm, f)
    plt.show()
