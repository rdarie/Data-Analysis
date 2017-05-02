from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold
import numpy as np
spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 500)
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 500)

#cValues = [1]
cValues = np.logspace(-3, 1, 10)

penalties = ['l2']
#penalties = ['l1','l2']

parameters = {'C': cValues, 'penalty' : penalties}

outputFileName = '/bestSpikeSVM_L.pickle'

trainSpikeMethod(spikeName, whichChans, SVC, skf, parameters, outputFileName)

print('Train SVM Linear Spikes DONE')
