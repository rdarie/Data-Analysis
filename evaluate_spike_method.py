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
whichChans = list(range(96))

dataName = '/saveSpikeRightLabeled.p'
dataFile = localDir + dataName
data = pd.read_pickle(dataFile)

features = data['spikes']
binCenters = data['binCenters']
spikeMat = data['spikeMat']
binWidth = data['binWidth']

modelName = '/bestSpikeLogReg.pickle'
modelFile = localDir + modelName
estimatorDict = pd.read_pickle(modelFile)
estimator = estimatorDict['estimator']
estimatorInfo = estimatorDict['info']

# get all columns of spikemat that aren't the labels
nonLabelChans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

nSamples = len(binCenters)
X = spikeMat[nonLabelChans]
y = spikeMat['LabelsNumeric']

# Poor man's test train split:
trainSize = 0.9
trainIdx = slice(None, int(trainSize * nSamples))
testIdx = slice(int(trainSize * nSamples) + 1, None)

estimator.fit(X[whichChans].iloc[trainIdx, :], y.iloc[trainIdx])
yHat = estimator.predict(X[whichChans].iloc[testIdx, :])

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
    #Plot the spikes
    fi = plotBinnedSpikes(X.iloc[testIdx], binCenters[testIdx], nonLabelChans, show = False)

    upMaskSpikes = (spikeMat['Labels'].iloc[testIdx] == 'Toe Up').values
    downMaskSpikes = (spikeMat['Labels'].iloc[testIdx] == 'Toe Down').values

    upMaskSpikesPredicted = (predictedLabels == 'Toe Up').values
    downMaskSpikesPredicted = (predictedLabels == 'Toe Down').values

    dummyVar = np.ones(binCenters[testIdx].shape[0]) * 1
    ax = fi.axes[0]
    ax.plot(binCenters[testIdx][upMaskSpikes], dummyVar[upMaskSpikes], 'ro')
    ax.plot(binCenters[testIdx][downMaskSpikes], dummyVar[downMaskSpikes] + 1, 'go')

    ax.plot(binCenters[testIdx][upMaskSpikesPredicted], dummyVar[upMaskSpikesPredicted] + .5, 'mo')
    ax.plot(binCenters[testIdx][downMaskSpikesPredicted], dummyVar[downMaskSpikesPredicted] + 1.5, 'co')

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
