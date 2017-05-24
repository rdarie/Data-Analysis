from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys, argparse, warnings
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpikeLDA.pickle')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'])
args = parser.parse_args()
argModel = args.model
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

estimator, estimatorInfo, _, _ = getEstimator(modelFileName)
modelName = getModelName(estimator)

whichChans = list(range(96))
X, y, trueLabels = getSpikeXY(dataNames, whichChans)
nSamples = y.shape[0]

dataFile = localDir + dataNames[-1]
data = pd.read_pickle(dataFile)
stepLen = data['binCenters'][1] - data['binCenters'][0]
binCenters = np.arange(nSamples) * stepLen
binWidth = data['binWidth']
spikeMat = data['spikeMat']
nonLabelChans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]
del(data)

suffix = modelName

if not os.path.isdir(localDir + '/' + modelName + '/'):
    os.makedirs(localDir + '/' + modelName + '/')

with open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+
    suffix + '.txt', 'w') as txtf:

    if isinstance(estimator, Pipeline):
        listOfEstimators = [x[1] for x in estimator.steps]
    else:
        listOfEstimators = [estimator]

    for step in listOfEstimators:
        txtf.write(getModelName(step) + '\n')
        for key, value in step.get_params().items():
            txtf.write(str(key) + '\n')
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    txtf.write(str(nested_key) + '\n')
                    txtf.write(str(nested_value) + '\n')
            else:
                txtf.write(str(value) + '\n')


    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 1)

    if __name__ == '__main__':

        yHat = cross_val_predict(estimator, X[whichChans], y,
            cv=skf, n_jobs = -1, pre_dispatch = 'n_jobs')

        labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
        labelsList = ['Neither', 'Toe Up', 'Toe Down']
        numericLabels = {v: k for k, v in labelsNumeric.items()}
        predictedLabels = pd.Series([numericLabels[x] for x in yHat])

        lS, yTrueLenient, yHatLenient = lenientScore(deepcopy(y), deepcopy(yHat), 0.05, 0.25,
            scoreFun = f1_score, average = 'macro')

        # Debugging
        # predictedLabels = pd.Series([numericLabels[x] for x in yHatLenient])

        txtf.write('\nLenient Score for '+ modelName + ' was:')
        txtf.write(str(lS))
        print('Lenient Score for '+ modelName + ' was:')
        print(lS)

        # Compute confusion matrix
        cnf_matrix = confusion_matrix(y, yHat, labels = [0,1,2])
        print("Normalized confusion matrix:")
        txtf.write("Normalized confusion matrix:\n")
        print(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis])
        txtf.write(str(cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]))

        # Compute F1 score
        f1Score = f1_score(y, yHat, average = 'macro')

        txtf.write('\nF1 Score for '+ modelName + ' was:')
        txtf.write(str(f1Score))
        print('F1 Score for '+ modelName + ' was:')
        print(f1Score)

        ROC_AUC = ROCAUC_ScoreFunction(estimator,
            X[whichChans], y)

        txtf.write('\nROC_AUC Score for '+ modelName + ' was:')
        txtf.write(str(ROC_AUC))
        print('ROC_AUC Score for '+ modelName + ' was:')
        print(ROC_AUC)

        txtReport = classification_report(y, yHat,
            labels = [0,1,2], target_names = labelsList)
        txtf.write(txtReport)

        plotting = True
        if plotting:
            #Plot the spikes
            fi = plotBinnedSpikes(X[whichChans], binCenters, nonLabelChans, show = False)

            upMaskSpikes = y == 1
            downMaskSpikes = y == 2

            upMaskSpikesPredicted = yHat == 1
            downMaskSpikesPredicted = yHat == 2

            dummyVar = np.ones(binCenters.shape[0]) * 1
            ax = fi.axes[0]
            ax.plot(binCenters[upMaskSpikes], dummyVar[upMaskSpikes], 'ro')
            ax.plot(binCenters[downMaskSpikes], dummyVar[downMaskSpikes] + 1, 'go')

            ax.plot(binCenters[upMaskSpikesPredicted], dummyVar[upMaskSpikesPredicted] + .5, 'mo')
            ax.plot(binCenters[downMaskSpikesPredicted], dummyVar[downMaskSpikesPredicted] + 1.5, 'co')

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
                warnings.warn("Unable to plot Validation Curve!", UserWarning)
                pass

            #plot a scatter matrix describing the performance:
            if hasattr(estimator, 'transform'):
                plotData = estimator.transform(X[whichChans])
                fiTr, ax = plt.subplots()
                if plotData.shape[1] == 2:
                    try:
                        ax.scatter(plotData[:, 0][y.values == 0],
                            plotData[:, 1][y.values == 0],
                            c = plt.cm.Paired(0.3), label = 'Neither')
                    except:
                        pass
                    try:
                        ax.scatter(plotData[:, 0][y.values == 1],
                            plotData[:, 1][y.values == 1],
                            c = plt.cm.Paired(0.6), label = 'Foot Off')
                    except:
                        pass
                    try:
                        ax.scatter(plotData[:, 0][y.values == 2],
                            plotData[:, 1][y.values == 2],
                            c = plt.cm.Paired(1), label = 'Foot Strike')
                    except:
                        pass
                else: # 1D
                    try:
                        ax.scatter(binCenters[y.values == 0],
                            plotData[:, 0][y.values == 0],
                            c = plt.cm.Paired(0.3), label = 'Neither')
                    except:
                        pass
                    try:
                        ax.scatter(binCenters[y.values == 1],
                            plotData[:, 0][y.values == 1],
                            c = plt.cm.Paired(0.6), label = 'Foot Off')
                    except:
                        pass
                    try:
                        ax.scatter(binCenters[y.values == 2],
                            plotData[:,0][y.values == 2],
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
                plotData = estimator.decision_function(X[whichChans])
                try:
                    ax.scatter(binCenters[y.values == 0],
                        plotData[:, 0][y.values == 0],
                        c = plt.cm.Paired(0.3), label = 'Neither')
                except:
                    pass
                try:
                    ax.scatter(binCenters[y.values == 1],
                        plotData[:, 0][y.values == 1],
                        c = plt.cm.Paired(0.6), label = 'Foot Off')
                except:
                    pass
                try:
                    ax.scatter(binCenters[y.values == 2],
                        plotData[:, 0][y.values == 2],
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
