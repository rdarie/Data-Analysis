from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys, argparse
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpikeLDA.pickle')
parser.add_argument('--nFeatures', default = 'all')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'])
args = parser.parse_args()
argModel = args.model
argNFeatures = args.nFeatures
argFile = args.file

nFeatures = float(argNFeatures) if argNFeatures != 'all' else None

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
dataNames = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
modelFileName = '/' + argModel

estimator, estimatorInfo, whichChans, maxFreq = getEstimator(modelFileName)
modelName = getModelName(estimator)
suffix = modelName

whichChans = list(range(96))
X, y, trueLabels = getSpikeXY(dataNames, whichChans)
skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
if isinstance(estimator, Pipeline) and 'downSampler' in estimator.named_steps.keys():
    trainX = estimator.named_steps['downSampler'].transform(X)
    trainY = y
else:
    trainX = X
    trainY = y

if isinstance(estimator, Pipeline) and 'scaler' in estimator.named_steps.keys():
    trainX = estimator.named_steps['scaler'].transform(trainX)

with open(localDir + '/' + modelName + '/spike_EstimatorInfo_'+
    suffix + '.txt', 'a') as txtf:

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

    if __name__ == '__main__':
        if isinstance(estimator, Pipeline):
            restOfPipeline = estimator.steps[-1][1]
            selector = fitRFECV(restOfPipeline, skf, trainX, trainY, nFeatures = nFeatures)
        else:
            selector = fitRFECV(estimator, skf, trainX, trainY, nFeatures = nFeatures)

        selectionOutcome = "\nOptimal number of features : %d" % selector.n_features_
        print(selectionOutcome)
        txtf.write(selectionOutcome)
        txtf.write('\nThese features were: ' + str(selector.support_) + '\n')
        txtf.write('\nThe ranking was: ' + str(selector.ranking_) + '\n')

        if argNFeatures == 'all':
            fi, ax = plt.subplots(1)

            plt.xlabel("Number of features selected")
            plt.ylabel("Cross validation score (ROC AUC)")
            ax.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
            #plt.show()
            plt.savefig(localDir + '/' + modelName + '/spike_FeatureSelection_'+ suffix + '.png')
            with open(localDir + '/' + modelName + '/spike_FeatureSelection_'+ suffix + '.pickle', 'wb') as f:
                pickle.dump(fi, f)

        featureSelector = FunctionTransformer(selectFromX, kw_args = {'support': selector.support_})

        if isinstance(estimator, Pipeline) and 'downSampler' in estimator.named_steps.keys():
            downSamplerIfPresent = [('downSampler', estimator.named_steps['downSampler'])]
        else:
            downSamplerIfPresent = []

        if isinstance(estimator, Pipeline) and 'scaler' in estimator.named_steps.keys():
            scalerIfPresent = [('scaler', estimator.named_steps['scaler'])]
        else:
            scalerIfPresent = []

        if isinstance(estimator, Pipeline):
            estimator = Pipeline(downSamplerIfPresent + scalerIfPresent +
                [('featureSelector', featureSelector)] +
                [(estimator.steps[-1][0], selector.estimator_)])
        else:
            estimator = Pipeline(downSamplerIfPresent + scalerIfPresent +
                [('featureSelector', featureSelector)] +
                [(modelName, selector.estimator_)])

        outputFileName = '/' + 'featureSelected_' + argModel
        bestSelector={'estimator' : estimator}

        with open(localDir + outputFileName, 'wb') as f:
            pickle.dump(bestSelector, f)
