from dataAnalysis.helperFunctions.helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
import numpy as np
spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 500)

SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 500, verbose = False, decision_function_shape = 'ovr', probability = False)
scaler = preprocessing.StandardScaler()

scaledSVC = Pipeline([('scaler', scaler), ('SVC', SVC)])
#cValues= [0.0025]
cValues = np.logspace(-2, 2, 15)

#gammaValues = ['auto']
gammaValues = np.logspace(-8,-3,20)

parameters = {'SVC__C': cValues, 'SVC__gamma': gammaValues}

outputFileName = '/bestSpikeSVM_RBF_Z.pickle'

trainSpikeMethod(spikeName, whichChans, scaledSVC, skf, parameters, outputFileName)

print('Train SVM RBF Z Spikes DONE')
