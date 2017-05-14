from helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
scaler = StandardScaler(copy = False)
SVC = svm.LinearSVC(class_weight = 'balanced', random_state = 500)
scaledSVC = Pipeline([('scaler', scaler),('SVCL', SVC)])
#cValues = [1]
cValues = np.logspace(-9, -1, 10)

penalties = ['l2']
#penalties = ['l1','l2']

parameters = {'SVCL__C': cValues, 'SVCL__penalty' : penalties}

outputFileName = '/bestSpikeSVMLZ.pickle'

trainSpikeMethod(dataName, whichChans, scaledSVC, skf, parameters,
    outputFileName, memPreallocate = 'n_jobs')
