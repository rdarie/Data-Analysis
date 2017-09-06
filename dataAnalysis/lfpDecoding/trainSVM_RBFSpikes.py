from dataAnalysis.helperFunctions.helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import numpy as np
spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 500)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 500, verbose = False, decision_function_shape = 'ovr', probability = False)

#cValues= [0.0025]
cValues = np.logspace(-5, -1, 25)

#gammaValues = ['auto']
gammaValues = np.logspace(-5,-1,25)

parameters = {'C': cValues, 'gamma': gammaValues}

outputFileName = '/bestSpikeSVM_RBF.pickle'

trainSpikeMethod(spikeName, whichChans, SVC, skf, parameters, outputFileName)

print('Train SVM RBF Spikes DONE')
