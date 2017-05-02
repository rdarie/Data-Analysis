from helper_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
dataName = '/saveSpectrumRightLabeled.p'

whichChans = range(1,97)
maxFreq = 200

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis()
downSampler = FunctionTransformer(freqDownSample)

downSampledLDA = Pipeline([('downSampler', downSampler), ('linDis', LDA)])
downSampleKWargs = [{'nChan' : 96, 'factor' : 1}, {'nChan' : 96, 'factor' : 5}, {'nChan' : 96, 'factor' : 10}]
componentCounts = [1,2]

parameters = { 'downSampler__kw_args' : downSampleKWargs, 'linDis__n_components' : componentCounts}
outputFileName = '/bestSpectrumLDA_DownSampled.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, downSampledLDA, skf, parameters, outputFileName)

print('Train LDA Spectrum DONE')
