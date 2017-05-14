from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import numpy as np
import argparse

parser = argparse.ArgumentParser()
#pdb.set_trace()
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev'])
args = parser.parse_args()
argFile = args.file

dataName = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
whichChans = range(96)


skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 500)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 500,
    cache_size = 250, verbose = 2, decision_function_shape = 'ovr', probability = False)

#cValues= [0.0025]
cValues = np.logspace(-9, -1, 10)

#gammaValues = ['auto']
gammaValues = np.logspace(-5,-2,10)

parameters = {'C': cValues, 'gamma': gammaValues}

outputFileName = '/bestSpikeSVMRBF.pickle'

trainSpikeMethod(dataName, whichChans, SVC, skf, parameters, outputFileName)

print('Train SVM RBF Spikes DONE')
