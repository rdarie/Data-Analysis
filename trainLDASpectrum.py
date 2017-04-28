from helper_functions import *
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

dataName = '/saveSpectrumRightLabeled.p'

whichChans = range(96)
maxFreq = 200

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis()

solvers = ['svd']
shrinkages = ['auto']
componentCounts = [None]

parameters = {'solver': solvers, 'n_components' : componentCounts}
outputFileName = '/bestSpectrumLDA.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, LDA, skf, parameters, outputFileName)
