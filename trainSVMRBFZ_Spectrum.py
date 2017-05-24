from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import itertools
import argparse

parser = argparse.ArgumentParser()
#pdb.set_trace()
parser.add_argument('--file', nargs='*',
    default = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.ns5'])
args = parser.parse_args()
argFile = args.file

dataName = ['/' + x.split('.')[0] + '_saveSpectrumLabeled.p' for x in argFile]

whichChans = list(range(1,97))
maxFreq = 500

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
scaler = StandardScaler(copy = False)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 250, verbose = 2, decision_function_shape = 'ovr', probability = False)

scaledSVC = Pipeline([('scaler', scaler),
    ('SVCRBF', SVC)])

cValues = np.logspace(-9, -1, 10)
gammaValues = np.logspace(-5,-2,10)

parameters = { 'SVCRBF__C': cValues,
    'SVCRBF__gamma': gammaValues}

outputFileName = '/bestSpectrumSVMRBFZ.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, scaledSVC,
    skf, parameters, outputFileName, memPreallocate = 'n_jobs')

print('Train SVMRBF Spectrum DONE')
