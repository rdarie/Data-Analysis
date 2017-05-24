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
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 500)

scaler = StandardScaler(copy = False)
scaledSVC = Pipeline([('scaler', scaler),
    ('LinearSVC', SVC)])

cValues = np.logspace(-7, 0, 10)

parameters = { 'LinearSVC__C': cValues}

outputFileName = '/bestSpectrumSVMLZ.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, scaledSVC,
    skf, parameters, outputFileName, memPreallocate = 'n_jobs')

print('Train SVMLZ Spectrum DONE')
