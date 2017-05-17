from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, matplotlib, argparse, string, warnings
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpectrumLDA_DownSampled.pickle')
parser.add_argument('--useRFE', nargs='*', default = '')
parser.add_argument('--file', nargs='*', default = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5','201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.ns5'])

args = parser.parse_args()
argModel = args.model
argUseRFE = args.useRFE
argFile = args.file

plotting = True
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
modelFileName = '/' + argModel
ns5Names = ['/' + x.split('.')[0] + '_saveSpectrumLabeled.p' for x in argFile]

estimator, estimatorInfo, whichChans, maxFreq = getEstimator(modelFileName)
modelName = getModelName(estimator)
print("Using estimator:")
print(estimator)

X, y, trueLabels = getSpectrumXY(ns5Names, whichChans, maxFreq)

ns5File = localDir + ns5Names[-1]
ns5Data = pd.read_pickle(ns5File)

origin = ns5Data['channel']['spectrum']['origin']
spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
fr = ns5Data['channel']['spectrum']['fr']
Fs = ns5Data['channel']['samp_per_s']
winLen = ns5Data['winLen']
stepLen = ns5Data['stepLen']
del(ns5Data)

suffix = modelName + '_winLen_' + str(winLen) + '_stepLen_' + str(stepLen) + '_from_' + origin
nSamples = y.shape[0]
# Poor man's test train split:
trainSize = 0.8
trainIdx = slice(None, int(trainSize * nSamples))
testIdx = slice(-(nSamples -int(trainSize * nSamples)), None)

if not os.path.isdir(localDir + '/' + modelName + '/'):
    os.makedirs(localDir + '/' + modelName + '/')

with open(localDir + '/' + modelName + '/spectrum_EstimatorInfo_'+ suffix + '.txt', 'w') as txtf:
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

    estimator.fit(X.iloc[trainIdx, :], y.iloc[trainIdx])

    if __name__ == '__main__':
        yHat = estimator.predict(X.iloc[testIdx, :])

        #get predicted labels
        labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
        labelsList = ['Neither', 'Toe Up', 'Toe Down']
        numericLabels = {v: k for k, v in labelsNumeric.items()}
        predictedLabels = pd.Series([numericLabels[x] for x in yHat])

        lS, yHatLenient = lenientScore(y.iloc[testIdx], yHat, 0.05,
            0.2, scoreFun = f1_score, average = 'macro')
        # Debugging
        predictedLabels = pd.Series([numericLabels[x] for x in yHatLenient])

        txtf.write('\nLenient Score for '+ modelName + ' was:')
        txtf.write(str(lS))
        print('Lenient Score for '+ modelName + ' was:')
        print(lS)

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

        txtReport = classification_report(y.iloc[testIdx], yHat,
            labels = [0,1,2], target_names = labelsList)
        txtf.write(txtReport)

        if plotting:
            #plot the spectrum
            upMaskSpectrum = (trueLabels.iloc[testIdx] == 'Toe Up').values
            downMaskSpectrum = (trueLabels.iloc[testIdx] == 'Toe Down').values
            dummyVar = np.ones(t[testIdx].shape[0]) * 1

            upMaskSpectrumPredicted = (predictedLabels == 'Toe Up').values
            downMaskSpectrumPredicted = (predictedLabels == 'Toe Down').values

            fi = plotSpectrum(spectrum[1].iloc[testIdx],
                Fs,
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
            plt.savefig(localDir + '/' + modelName + '/spectrum_Plot_' + suffix  + '.png')
            with open(localDir + '/' + modelName + '/spectrum_Plot_' + suffix  + '.pickle', 'wb') as f:
                pickle.dump(fi, f)

            if isinstance(estimator, Pipeline):
                if 'downSampler' in estimator.named_steps.keys():
                    downSampler = estimator.named_steps['downSampler']
                    reducedX = downSampler.transform(X,y)
                    fiDs = plotFeature(reducedX, y)
                    plt.tight_layout()
                    plt.savefig(localDir + '/' + modelName + '/spectrum_DownSampledFeatures_' + suffix  + '.png')
                    with open(localDir + '/' + modelName + '/spectrum_DownSampledFeatures_' + suffix  + '.pickle', 'wb') as f:
                        pickle.dump(fiDs, f)

            # Plot normalized confusion matrix
            fiCm = plotConfusionMatrix(cnf_matrix, classes = labelsList, normalize=True,
                title='Normalized confusion matrix')
            plt.tight_layout()
            plt.savefig(localDir + '/' + modelName + '/spectrum_ConfusionMatrix_' + suffix  + '.png')
            with open(localDir + '/' + modelName + '/spectrum_ConfusionMatrix_' + suffix  + '.pickle', 'wb') as f:
                pickle.dump(fiCm, f)

            # Plot a validation Curve
            try:
                fiVC = plotValidationCurve(estimator, estimatorInfo)
                plt.savefig(localDir + '/' + modelName + '/spectrumValidationCurve' + suffix  + '.png')
                with open(localDir + '/' + modelName + '/spectrumValidationCurve' + suffix  + '.pickle', 'wb') as f:
                    pickle.dump(fiVC, f)
            except:
                warnings.warn("Unable to plot Validation Curve!", UserWarning)
                pass

            #plot a scatter matrix describing the performance:
            if hasattr(estimator, 'transform'):
                plotData = estimator.transform(X.iloc[testIdx])
                fiTr, ax = plt.subplots()
                if plotData.shape[1] == 2: #2D
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
                plt.savefig(localDir + '/' + modelName + '/spectrum_Transformed_' + suffix  + '.png')
                plt.show(block = False)
                with open(localDir + '/' + modelName + '/spectrum_Transformed_' + suffix  + '.pickle', 'wb') as f:
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

                plt.savefig(localDir + '/' + modelName + '/spectrum_DecisionBoundary_' + suffix  + '.png')
                plt.show(block = False)
                with open(localDir + '/' + modelName + '/spectrum_DecisionBoundary_' + suffix  + '.pickle', 'wb') as f:
                    pickle.dump(fiDb, f)
            plt.show()
