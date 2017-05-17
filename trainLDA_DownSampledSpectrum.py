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
downSampler = FunctionTransformer(freqDownSample)

downSampledLDA = Pipeline([('downSampler', downSampler),
    ('linearDiscriminantAnalysis', LDA)])

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

### Debugging:
"""
keepChans = [ sorted(random.sample(range(len(whichChans)),
    m.floor(len(whichChans) / nSubsampled))) for nSubsampled in [15, 30] ]

bands = []
for x in range(1,3):
    bands = bands + list(itertools.combinations([
        (8,15),
        (400,500)
        ],x))

"""

downSampleKWargs = [{'whichChans' : whichChans, 'strategy': 'bands',
    'maxFreq' : maxFreq,
    'bands' : x, 'keepChans': y} for x,y in itertools.product(bands,keepChans)]

shrinkages = [0.1 * float(x) for x in range(10)] + ['auto']

parameters = { 'downSampler__kw_args' : downSampleKWargs,
    'linearDiscriminantAnalysis__shrinkage' : shrinkages}

outputFileName = '/bestSpectrumLDA_DownSampled.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, downSampledLDA,
    skf, parameters, outputFileName, memPreallocate = 'n_jobs')

print('Train LDA DownSampled Spectrum DONE')
