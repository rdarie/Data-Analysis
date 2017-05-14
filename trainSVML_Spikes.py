from helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.pipeline import Pipeline
import numpy as np

parser = argparse.ArgumentParser()
#pdb.set_trace()
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev'])
args = parser.parse_args()
argFile = args.file

dataName = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
whichChans = range(96)


skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 500)

#cValues = [1]
cValues = np.logspace(-9, -1, 20)

penalties = ['l2']
#penalties = ['l1','l2']

parameters = {'C': cValues, 'penalty' : penalties}

outputFileName = '/bestSpikeSVML.pickle'

trainSpikeMethod(dataName, whichChans, SVC, skf, parameters,
    outputFileName, memPreallocate = 'n_jobs')
