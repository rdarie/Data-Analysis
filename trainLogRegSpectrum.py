from helper_functions import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import numpy as np

dataName = '/saveSpectrumRightLabeled.p'

whichChans = range(1,97)
maxFreq = 200

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
logReg = LogisticRegression(class_weight = 'balanced', max_iter = 500)

#cValues=np.logspace(-2,5,10)
cValues = np.logspace(-3, 1, 10)
solvers = ['liblinear']
penalties = ['l2']

parameters = {'C': cValues, 'solver': solvers, 'penalty': penalties}
outputFileName = '/bestSpectrumLogReg.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, logReg, skf, parameters, outputFileName)
print('Train LogReg Spectrum DONE')
