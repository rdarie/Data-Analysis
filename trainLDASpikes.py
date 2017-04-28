from helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis()

solvers = ['svd']
shrinkages = ['auto']
componentCounts = [None]

parameters = {'solver': solvers, 'n_components' : componentCounts}
outputFileName = '/bestSpikeLDA.pickle'

trainSpikeMethod(spikeName, whichChans, LDA, skf, parameters, outputFileName)
