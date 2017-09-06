from dataAnalysis.helperFunctions.helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
import numpy as np
spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
logReg = LogisticRegression(class_weight = 'balanced', max_iter = 500)

cValues = np.logspace(-3, 1, 10)
solvers = ['liblinear']
penalties = ['l2']
parameters = {'C': cValues,'penalty': penalties, 'solver' : solvers}

outputFileName = '/bestSpikeLogReg.pickle'

trainSpikeMethod(spikeName, whichChans, logReg, skf, parameters, outputFileName)
