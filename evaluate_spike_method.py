from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys
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

try:
    modelName = '/' + sys.argv[1]
except:
    modelName = '/bestSpikeSVM_RBF_Z.pickle'

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
print('F1 Score for '+ estimator.__str__()[:3] + ' was:')
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

    plt.tight_layout()
    plt.savefig(localDir + '/spike'+ estimator.__str__()[:3] + 'Plot.png')

    # Plot normalized confusion matrix
    fiCm = plotConfusionMatrix(cnf_matrix, classes = labelsNumeric.keys(), normalize=True,
                          title='Normalized confusion matrix')

    plt.tight_layout()
    plt.savefig(localDir + '/spike'+ estimator.__str__()[:3] + 'ConfusionMatrix.png')

    # Plot a validation Curve
    fiVC = plotValidationCurve(estimator, estimatorInfo)

    plt.savefig(localDir + '/spike'+ estimator.__str__()[:3] + 'ValidationCurve.png')

    #plot a scatter matrix describing the performance:
    if hasattr(estimator, 'transform'):
        plotData = estimator.transform(X.iloc[testIdx])
        fiTr, ax = plt.subplots()
        if plotData.shape[1] == 2:
            try:
                ax.scatter(plotData[:, 0][y.iloc[testIdx].values == 0],
                    plotData[:, 1][y.iloc[testIdx].values == 0],
                    c = plt.cm.Paired(0.3), label = 'Neither')
            except:
                pass
            try:
                ax.scatter(plotData[:, 0][y.iloc[testIdx].values == 1],
                    plotData[:, 1][y.iloc[testIdx].values == 1],
                    c = plt.cm.Paired(0.6), label = 'Foot Off')
            except:
                pass
            try:
                ax.scatter(plotData[:, 0][y.iloc[testIdx].values == 2],
                    plotData[:, 1][y.iloc[testIdx].values == 2],
                    c = plt.cm.Paired(1), label = 'Foot Strike')
            except:
                pass
        else: # 1D
            try:
                ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 0],
                    plotData[:, 0][y.iloc[testIdx].values == 0],
                    c = plt.cm.Paired(0.3), label = 'Neither')
            except:
                pass
            try:
                ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 1],
                    plotData[:, 0][y.iloc[testIdx].values == 1],
                    c = plt.cm.Paired(0.6), label = 'Foot Off')
            except:
                pass
            try:
                ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 2],
                    plotData[:,0][y.iloc[testIdx].values == 2],
                    c = plt.cm.Paired(1), label = 'Foot Strike')
            except:
                pass

        plt.legend(markerscale=2, scatterpoints=1)
        ax.set_title('Method Transform')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.tight_layout()
        plt.savefig(localDir + '/spike'+ estimator.__str__()[:3] + 'TransformedPlot.png')

        plt.show()
        with open(localDir + '/spike'+ estimator.__str__()[:3] + 'TransformedPlot.pickle', 'wb') as f:
            pickle.dump(fiTr, f)

    if hasattr(estimator, 'decision_function'):
        fiDb, ax = plt.subplots()
        plotData = estimator.decision_function(X.iloc[testIdx])
        try:
            ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 0],
                plotData[:, 0][y.iloc[testIdx].values == 0],
                c = plt.cm.Paired(0.3), label = 'Neither')
        except:
            pass
        try:
            ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 1],
                plotData[:, 0][y.iloc[testIdx].values == 1],
                c = plt.cm.Paired(0.6), label = 'Foot Off')
        except:
            pass
        try:
            ax.scatter(binCenters[testIdx][y.iloc[testIdx].values == 2],
                plotData[:, 0][y.iloc[testIdx].values == 2],
                c = plt.cm.Paired(0.1), label = 'Foot Strike')
        except:
            pass
        ax.set_xlabel('Time (sec)')
        ax.set_title('Distance from Neither Boundary')
        #ax.set_yticks(())
        plt.legend(markerscale=2, scatterpoints=1)
        plt.tight_layout()
        plt.savefig(localDir + '/spike'+ estimator.__str__()[:3] + 'DecisionBoundaryPlot.png')

        plt.show()
        with open(localDir + '/spike'+ estimator.__str__()[:3] + 'DecisionBoundaryPlot.pickle', 'wb') as f:
            pickle.dump(fiDb, f)

    figDic = {'spectrum': fi, 'confusion': fiCm}

    with open(localDir + '/spike'+ estimator.__str__()[:3] + 'Plot.pickle', 'wb') as f:
        pickle.dump(fi, f)
    with open(localDir + '/spike'+ estimator.__str__()[:3] + 'ConfusionMatrix.pickle', 'wb') as f:
        pickle.dump(fiCm, f)
    with open(localDir + '/spike'+ estimator.__str__()[:3] + 'ValidationCurve.pickle', 'wb') as f:
        pickle.dump(fiVC, f)
    plt.show()
