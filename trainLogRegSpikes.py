from helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression

spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
logReg = LogisticRegression(class_weight = 'balanced', max_iter = 500)

cValues= [1]
solvers = ['liblinear']
penalties = ['l2']
parameters = {'C': cValues,'penalty': penalties, 'solver' : solvers}

outputFileName = '/bestSpikeLogReg.pickle'

trainSpikeMethod(spikeName, whichChans, logReg, skf, parameters, outputFileName)
