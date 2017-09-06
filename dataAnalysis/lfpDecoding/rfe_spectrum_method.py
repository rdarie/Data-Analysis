from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, matplotlib, argparse, string
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpectrumLDA_DownSampled.pickle')
parser.add_argument('--nFeatures', default = 'all')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.ns5'])

args = parser.parse_args()
argModel = args.model
argNFeatures = args.nFeatures
argFile = args.file

nFeatures = float(argNFeatures) if argNFeatures != 'all' else None

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
modelFileName = '/' + argModel
ns5Names = ['/' + x.split('.')[0] + '_saveSpectrumLabeled.p' for x in argFile]

modelFile = localDir + modelFileName
estimatorDict = pd.read_pickle(modelFile)

estimator = estimatorDict['estimator']
estimatorInfo = estimatorDict['info']
whichChans = estimatorDict['whichChans']
maxFreq = estimatorDict['maxFreq']
modelName = getModelName(estimator)
print("Using estimator:")
print(estimator)

X, y, trueLabels = getSpectrumXY(ns5Names, whichChans, maxFreq)

ns5File = localDir + ns5Names[-1]
ns5Data = pd.read_pickle(ns5File)

origin = ns5Data['channel']['spectrum']['origin']
#spectrum = ns5Data['channel']['spectrum']['PSD']
#t = ns5Data['channel']['spectrum']['t']
#fr = ns5Data['channel']['spectrum']['fr']
#Fs = ns5Data['channel']['samp_per_s']
winLen = ns5Data['winLen']
stepLen = ns5Data['stepLen']
del(ns5Data)

suffix = modelName + '_winLen_' + str(winLen) + '_stepLen_' + str(stepLen) + '_from_' + origin

if not os.path.isdir(localDir + '/' + modelName + '/'):
    os.makedirs(localDir + '/' + modelName + '/')

with open(localDir + '/' + modelName + '/spectrum_EstimatorInfo_'+ suffix +
    '.txt', 'a') as txtf:

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

    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)

    if __name__ == '__main__':
        if isinstance(estimator, Pipeline) and 'downSampler' in estimator.named_steps.keys():
            trainX = estimator.named_steps['downSampler'].transform(X)
            trainY = y
        else:
            trainX = X
            trainY = y

        if isinstance(estimator, Pipeline) and 'scaler' in estimator.named_steps.keys():
            trainX = estimator.named_steps['scaler'].transform(trainX)

        if isinstance(estimator, Pipeline):
            restOfPipeline = estimator.steps[-1][1]
            selector = fitRFECV(restOfPipeline, skf, trainX,
                trainY, nFeatures = nFeatures)
        else:
            selector = fitRFECV(estimator, skf, trainX,
                trainY, nFeatures = nFeatures)

        outputFileName = '/' + 'featureSelected_' +argModel

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
            plt.savefig(localDir + '/' + modelName + '/spectrum_FeatureSelection_'+ suffix + '.png')
            with open(localDir + '/' + modelName + '/spectrum_FeatureSelection_'+ suffix + '.pickle', 'wb') as f:
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

        bestSelector={'estimator' : estimator,
            'whichChans' : whichChans, 'maxFreq' : maxFreq}
        with open(localDir + outputFileName, 'wb') as f:
            pickle.dump(bestSelector, f)
