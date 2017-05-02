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
downSampler = FunctionTransformer(downSampleFreq, kw_args={'factor': 10})

downSampledLDA = Pipeline([('downSampler', downSampler)])

solvers = ['svd']
shrinkages = ['auto']
componentCounts = [1,2]

parameters = {'solver': solvers, 'n_components' : componentCounts}
outputFileName = '/bestSpectrumLDA.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, LDA, skf, parameters, outputFileName)

print('Train LDA Spectrum DONE')
