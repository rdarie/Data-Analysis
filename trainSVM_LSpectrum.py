from helper_functions import *
from sklearn import svm
from sklearn.model_selection import StratifiedKFold

dataName = '/saveSpectrumRightLabeled.p'

whichChans = range(96)
maxFreq = 200

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 1)

cValues = [1]
penalties = ['l2']
parameters = {'C': cValues, 'penalty' : penalties}
outputFileName = '/bestSpectrumSVM_L.pickle'

trainSpectralMethod(dataName, whichChans, maxFreq, SVC, skf, parameters, outputFileName)

print('Train SVM Linear Spectrum DONE')
