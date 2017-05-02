from helper_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import itertools
dataName = '/saveSpectrumRightLabeled.p'

whichChans = list(range(1,97))
maxFreq = 500

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis()
downSampler = FunctionTransformer(freqDownSample)

downSampledLDA = Pipeline([('downSampler', downSampler), ('linDis', LDA)])
keepChans = [ sorted(random.sample(range(len(whichChans)),
    m.floor(len(whichChans) / nSubsampled))) for nSubsampled in [1, 2, 5, 10, 20, 30] ]
downSampleKWargs = [{'whichChans' : whichChans, 'freqFactor' : x, 'keepChans': y} for x,y in itertools.product([1, 2, 5, 10, 20],keepChans)]
#componentCounts = [1,2]

#parameters = { 'downSampler__kw_args' : downSampleKWargs, 'linDis__n_components' : componentCounts}
parameters = { 'downSampler__kw_args' : downSampleKWargs}

outputFileName = '/bestSpectrumLDA_DownSampled.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, downSampledLDA, skf, parameters, outputFileName)

print('Train LDA DownSampled Spectrum DONE')
