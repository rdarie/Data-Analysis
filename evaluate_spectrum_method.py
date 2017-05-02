from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, matplotlib
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
try:
    modelName = '/' + sys.argv[1]
except:
    modelName = '/bestSpectrumLDA_DownSampled.pickle'

ns5Name = '/saveSpectrumRightLabeled.p'

ns5File = localDir + ns5Name
ns5Data = pd.read_pickle(ns5File)

spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
fr = ns5Data['channel']['spectrum']['fr']
labels = ns5Data['channel']['spectrum']['LabelsNumeric']

modelFile = localDir + modelName
estimatorDict = pd.read_pickle(modelFile)
estimator = estimatorDict['estimator']
estimatorInfo = estimatorDict['info']

nSamples = len(t)
y = labels

whichChans = estimatorDict['whichChans']
maxFreq = estimatorDict['maxFreq']

whichFreqs = ns5Data['channel']['spectrum']['fr'] < maxFreq
flatSpectrum = spectrum[whichChans, :, whichFreqs].transpose(1, 0, 2).to_frame().transpose()
X = flatSpectrum

# Poor man's test train split:
trainSize = 0.95
trainIdx = slice(None, int(trainSize * nSamples))
testIdx = slice(int(trainSize * nSamples) + 1, None)

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
f1Score = f1_score(y.iloc[testIdx], yHat, average = 'macro')
print('F1 Score for '+ estimator.__str__()[:11] + ' was:')
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

    plt.tight_layout()
    plt.savefig(localDir + '/spectrum'+ estimator.__str__()[:11] + 'Plot.png')

    # Plot normalized confusion matrix
    fiCm = plotConfusionMatrix(cnf_matrix, classes = labelsNumeric.keys(), normalize=True,
                          title='Normalized confusion matrix')

    plt.tight_layout()
    plt.savefig(localDir + '/spectrum'+ estimator.__str__()[:11] + 'ConfusionMatrix.png')

    # Plot a validation Curve
    fiVC = plotValidationCurve(estimator, estimatorInfo)
    #plt.tight_layout()
    plt.savefig(localDir + '/spectrum'+ estimator.__str__()[:11] + 'ValidationCurve.png')

    figDic = {'spectrum': fi, 'confusion': fiCm, 'validation': fiVC}

    with open(localDir + '/spectrum'+ estimator.__str__()[:11] + 'Plot.pickle', 'wb') as f:
        pickle.dump(fi, f)
    with open(localDir + '/spectrum'+ estimator.__str__()[:11] + 'ConfusionMatrix.pickle', 'wb') as f:
        pickle.dump(fiCm, f)
    with open(localDir + '/spectrum'+ estimator.__str__()[:11] + 'ValidationCurve.pickle', 'wb') as f:
        pickle.dump(fiVC, f)
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
                ax.scatter(t[testIdx][y.iloc[testIdx].values == 0],
                    plotData[:, 0][y.iloc[testIdx].values == 0],
                    c = plt.cm.Paired(0.3), label = 'Neither')
            except:
                pass
            try:
                ax.scatter(t[testIdx][y.iloc[testIdx].values == 1],
                    plotData[:, 0][y.iloc[testIdx].values == 1],
                    c = plt.cm.Paired(0.6), label = 'Foot Off')
            except:
                pass
            try:
                ax.scatter(t[testIdx][y.iloc[testIdx].values == 2],
                    plotData[:, 0][y.iloc[testIdx].values == 2],
                    c = plt.cm.Paired(1), label = 'Foot Strike')
            except:
                pass

        plt.legend(markerscale=2, scatterpoints=1)
        ax.set_title('Method Transform')
        ax.set_xticks(())
        ax.set_yticks(())
        plt.tight_layout()
        plt.savefig(localDir + '/spectrum'+ estimator.__str__()[:11] + 'TransformedPlot.png')
        plt.show(block = False)
        with open(localDir + '/spectrum'+ estimator.__str__()[:11] + 'TransformedPlot.pickle', 'wb') as f:
            pickle.dump(fiTr, f)

    if hasattr(estimator, 'decision_function'):
        fiDb, ax = plt.subplots()
        plotData = estimator.decision_function(X.iloc[testIdx])

        try:
            ax.scatter(t[testIdx][y.iloc[testIdx].values == 0],
                plotData[:, 0][y.iloc[testIdx].values == 0],
                c = plt.cm.Paired(0.3), label = 'Neither')
        except:
            pass
        try:
            ax.scatter(t[testIdx][y.iloc[testIdx].values == 1],
                plotData[:, 0][y.iloc[testIdx].values == 1],
                c = plt.cm.Paired(0.6), label = 'Foot Off')
        except:
            pass
        try:
            ax.scatter(t[testIdx][y.iloc[testIdx].values == 2],
                plotData[:, 0][y.iloc[testIdx].values == 2],
                c = plt.cm.Paired(0.1), label = 'Foot on')
        except:
            pass

        ax.set_xlabel('Time (sec)')
        ax.set_title('Distance from Neither Boundary')
        #ax.set_yticks(())
        plt.legend(markerscale=2, scatterpoints=1)
        plt.tight_layout()
        plt.savefig(localDir + '/spectrum'+ estimator.__str__()[:11] + 'DecisionBoundaryPlot.png')
        plt.show(block = False)
        with open(localDir + '/spectrum'+ estimator.__str__()[:11] + 'DecisionBoundaryPlot.pickle', 'wb') as f:
            pickle.dump(fiDb, f)
    plt.show()
