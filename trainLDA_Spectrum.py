from helper_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
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

skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis(n_components = 2, solver = 'eigen',
    shrinkage = 'auto')

#shrinkages = [0.1 * float(x) for x in range(10)] + ['auto']
shrinkages = ['auto']
parameters = { 'n_components' : [2],
    'shrinkage' : shrinkages}

outputFileName = '/bestSpectrumLDA.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, LDA,
    skf, parameters, outputFileName, memPreallocate = 'n_jobs')

print('Train LDA Spectrum DONE')
