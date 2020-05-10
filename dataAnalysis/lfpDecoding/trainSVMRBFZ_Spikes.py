from dataAnalysis.helperFunctions.helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev'])
args = parser.parse_args()
argFile = args.file

dataName = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
whichChans = range(96)

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 500)
scaler = StandardScaler(copy = False)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 1000, verbose = 2, decision_function_shape = 'ovr', probability = False)

scaledSVC = Pipeline([('scaler', scaler),('SVMRBF', SVC)])

#cValues= [0.0025]
cValues = np.logspace(-9, -1, 10)

#gammaValues = ['auto']
gammaValues = np.logspace(-10,-2,10)

parameters = {'SVMRBF__C': cValues, 'SVMRBF__gamma': gammaValues}

outputFileName = '/bestSpikeSVMRBFZ.pickle'

trainSpikeMethod(dataName, whichChans, scaledSVC, skf, parameters,
    outputFileName, memPreallocate = '2 * n_jobs')

print('Train SVMRBFZ Spikes DONE')
