from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

spikeName = '/saveSpikeRightLabeled.p'
whichChans = list(range(96))

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 1)

cValues= [1]
penalties = ['l2']
parameters = {'C': cValues, 'penalty' : penalties}

outputFileName = '/bestSpikeSVM_L.pickle'

trainSpikeMethod(spikeName, whichChans, SVC, skf, parameters, outputFileName)

print('Train SVM Linear Spikes DONE')
