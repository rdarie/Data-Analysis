from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys, argparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpikeSVML.pickle')
parser.add_argument('--useRFE', nargs='*', default = '')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'])
args = parser.parse_args()
argModel = args.model
argUseRFE = args.useRFE
argFile = args.file
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
dataNames = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
modelFileName = '/' + argModel

modelFile = localDir + modelFileName
estimatorDict = pd.read_pickle(modelFile)
estimator = estimatorDict['estimator']
estimatorInfo = estimatorDict['info']
modelName = getModelName(estimator)

whichChans = list(range(96))
X, y, trueLabels = pd.DataFrame(), pd.Series(), pd.Series()

for idx, dataName in enumerate(dataNames):
    #get all columns of spikemat that aren't the labels
    dataFile = localDir + dataName
    data = pd.read_pickle(dataFile)

    spikeMat = data['spikeMat']
    nonLabelChans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

    X = pd.concat((X,spikeMat[nonLabelChans]))
    y = pd.concat((y,spikeMat['LabelsNumeric']))
    trueLabels = pd.concat((trueLabels, spikeMat['Labels']))

    if idx == len(dataNames) - 1:
        binCenters = data['binCenters']
        binWidth = data['binWidth']

suffix = modelName

if not os.path.isdir(localDir + '/' + modelName + '/'):
    os.makedirs(localDir + '/' + modelName + '/')

nSamples = y.shape[0]
# Poor man's test train split:
trainSize = 0.8
trainIdx = slice(None, int(trainSize * nSamples))
testIdx = slice(-(nSamples -int(trainSize * nSamples)), None)

with open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+ suffix + '.txt', 'w') as txtf:
    if type(estimator) == Pipeline:
        listOfEstimators = [x[1] for x in estimator.steps]
    else:
        listOfEstimators = [estimator]

    for step in listOfEstimators:
        txtf.write(getModelName(step) + '\n')

    useRFE = bool(argUseRFE)
    if useRFE:
        skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
        if __name__ == '__main__':
            if estimator is Pipeline and 'downSampler' in estimator.named_steps.keys():
                trainX = estimator.named_steps['downSampler'].transform(X.iloc[trainIdx, :])
                trainY = y.iloc[trainIdx]
            else:
                trainX = X.iloc[trainIdx, :]
                trainY = y.iloc[trainIdx]

            if estimator is Pipeline and 'scaler' in estimator.named_steps.keys():
                trainX = estimator.named_steps['scaler'].transform(trainX)
            #restOfPipeline = Pipeline([(name, estimator) for name, estimator in estimator.steps if not name == 'downSampler'])
            if estimator is Pipeline:
                restOfPipeline = estimator.steps[-1][1]
                selector = fitRFECV(restOfPipeline, skf, trainX, trainY)
            else:
                selector = fitRFECV(estimator, skf, trainX, trainY)

            outputFileName = '/' + 'featureSelected_' + argModel
            bestSelector={'estimator' : selector}

            with open(localDir + outputFileName, 'wb') as f:
                pickle.dump(bestSelector, f)


            selectionOutcome = "Optimal number of features : %d" % selector.n_features_
            print(selectionOutcome)
            txtf.write(selectionOutcome)fi, ax = plt.subplots(1)

            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (ROC AUC)")
            ax.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
            #plt.show()
            plt.savefig(localDir + '/' + modelName + '/spike_FeatureSelection_'+ suffix + '.png')
            with open(localDir + '/' + modelName + '/spike_FeatureSelection_'+ suffix + '.pickle', 'wb') as f:
                pickle.dump(fi, f)

            featureSelector = FunctionTransformer(selectFromX, kw_args = {'support': selector.support_})

            if estimator is Pipeline and 'downSampler' in estimator.named_steps.keys():
                downSamplerIfPresent = [('downSampler', estimator.named_steps['downSampler'])]
            else:
                downSamplerIfPresent = []

            if estimator is Pipeline and 'scaler' in estimator.named_steps.keys():
                scalerIfPresent = [('scaler', estimator.named_steps['scaler'])]
            else:
                scalerIfPresent = []

            estimator = Pipeline(downSamplerIfPresent + scalerIfPresent +
                [('featureSelector', featureSelector)] +
                [(estimator.steps[-1][0], selector.estimator_)])

    else:
        estimator.fit(X[whichChans].iloc[trainIdx, :], y.iloc[trainIdx])

    if __name__ == '__main__':
        yHat = estimator.predict(X[whichChans].iloc[testIdx, :])

        labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
        labelsList = ['Neither', 'Toe Up', 'Toe Down']
        numericLabels = {v: k for k, v in labelsNumeric.items()}
        predictedLabels = pd.Series([numericLabels[x] for x in yHat])

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y.iloc[testIdx], yHat, labels = [0,1,2])
        print("Normalized confusion matrix:")
        txtf.write("Normalized confusion matrix:\n")
        print(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])
        txtf.write(str(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]))

        # Compute F1 score
        f1Score = f1_score(y.iloc[testIdx], yHat, average = 'macro')

        txtf.write('\nF1 Score for '+ modelName + ' was:')
        txtf.write(str(f1Score))
        print('F1 Score for '+ modelName + ' was:')
        print(f1Score)

        ROC_AUC = ROCAUC_ScoreFunction(estimator,
            X.iloc[testIdx, :], y.iloc[testIdx])

        txtf.write('\nROC_AUC Score for '+ modelName + ' was:')
        txtf.write(str(ROC_AUC))
        print('ROC_AUC Score for '+ modelName + ' was:')
        print(ROC_AUC)

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
            plt.savefig(localDir + '/' + modelName + '/spike_Plot_'+ suffix + '.png')
            with open(localDir + '/' + modelName + '/spike_Plot_'+ suffix + '.pickle', 'wb') as f:
                pickle.dump(fi, f)

            # Plot normalized confusion matrix
            fiCm = plotConfusionMatrix(cnf_matrix, classes = labelsList, normalize=True,
                                  title='Normalized confusion matrix')

            plt.tight_layout()
            plt.savefig(localDir + '/' + modelName + '/spike_ConfusionMatrix_'+ suffix + '.png')
            with open(localDir + '/' + modelName + '/spike_ConfusionMatrix_'+ suffix + '.pickle', 'wb') as f:
                pickle.dump(fiCm, f)

            # Plot a validation Curve
            try:
                fiVC = plotValidationCurve(estimator, estimatorInfo)
                plt.savefig(localDir + '/' + modelName + '/spike_ValidationCurve_'+ suffix + '.png')
                with open(localDir + '/' + modelName + '/spike_ValidationCurve_'+ suffix + '.pickle', 'wb') as f:
                    pickle.dump(fiVC, f)
            except:
                pass

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
                plt.savefig(localDir + '/' + modelName + '/spike_TransformedPlot_'+ suffix + '.png')
                with open(localDir + '/' + modelName + '/spike_TransformedPlot_'+ suffix + '.pickle', 'wb') as f:
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
                plt.savefig(localDir + '/' + modelName +
                    '/spike_DecisionBoundaryPlot_'+ suffix + '.png')
                with open(localDir + '/' + modelName +
                    '/spike_DecisionBoundaryPlot_'+ suffix + '.pickle', 'wb') as f:
                    pickle.dump(fiDb, f)

            figDic = {'spectrum': fi, 'confusion': fiCm}

            plt.show()
