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
downSampler = FunctionTransformer(freqDownSample)

downSampledSVC = Pipeline([('downSampler', downSampler),('scaler', scaler),('SVCRBF', SVC)])

keepChans = [ sorted(random.sample(range(len(whichChans)),
    m.floor(len(whichChans) / nSubsampled))) for nSubsampled in [1] ]

bands = []
for x in range(1,8):
    bands = bands + list(itertools.combinations([
        (8,15),
        (15,35),
        (35,100),
        (100,200),
        (200,300),
        (300,400),
        (400,500)
        ],x))

cValues = np.logspace(-9, -1, 10)

#downSampleKWargs = [{'whichChans' : whichChans, 'freqFactor' : x, 'keepChans': y} for x,y in itertools.product([1, 5, 10, 15],keepChans)]
downSampleKWargs = [{'whichChans' : whichChans, 'strategy': 'bands',
    'maxFreq' : maxFreq, 'bands' : x, 'keepChans': y}
    for x,y in itertools.product(bands,keepChans)]

#componentCounts = [1,2]

#parameters = { 'downSampler__kw_args' : downSampleKWargs, 'linDis__n_components' : componentCounts}
parameters = { 'downSampler__kw_args' : downSampleKWargs, 'SVCL__C': cValues}

outputFileName = '/bestSpectrumSVML_DownSampled.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, downSampledSVC,
    skf, parameters, outputFileName, memPreallocate = 'n_jobs')

print('Train SVC DownSampled Spectrum DONE')
