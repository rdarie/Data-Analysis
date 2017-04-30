from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
SVC = svm.SVC(kernel = 'rbf', class_weight = 'balanced', random_state = 1)

cValues= [1]
parameters = {'C': cValues}

outputFileName = '/bestSpikeSVM_RBF.pickle'

trainSpikeMethod(spikeName, whichChans, SVC, skf, parameters, outputFileName)

print('Train SVM RBF Spikes DONE')
