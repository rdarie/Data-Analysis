from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import numpy as np
dataName = '/saveSpectrumRightLabeled.p'

#whichChans = range(1,97)
whichChans = [0, 25, 50, 75]
maxFreq = 500

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 250, verbose = True, decision_function_shape = 'ovr')

#cValues= [0.0025]
cValues = np.logspace(-3, 1, 10)

#gammaValues = ['auto']
# See https://stats.stackexchange.com/questions/125353/output-of-scikit-svm-in-multiclass-classification-always-gives-same-label
# for discussion of gamma tuning
gammaValues = np.logspace(-5,-3,20)

parameters = {'C': cValues, 'gamma': gammaValues}

outputFileName = '/bestSpectrumSVM_RBF.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, SVC, skf, parameters, outputFileName)

print('Train SVM RBF Spectrum DONE')
