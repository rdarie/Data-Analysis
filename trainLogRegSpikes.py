from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression

# Plotting options
font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

spikeName = '/saveSpikeRightLabeled.p'
spikeFile = localDir + spikeName
spikeData = pd.read_pickle(spikeFile)

spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']

# get all columns of spikemat that aren't the labels
chans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

X = spikeMat[chans]
y = spikeMat['LabelsNumeric']

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
logReg = LogisticRegression(class_weight = 'balanced', max_iter = 500)

cValues=np.logspace(-1,4,3)
solvers = ['sag', 'liblinear']
penalties = ['l2']

logGrid=GridSearchCV(logReg,{'C': cValues,'penalty': penalties, 'solver' : solvers}, scoring = 'f1_weighted', cv = skf, n_jobs = -1, verbose = 4)

if __name__ == '__main__':
    logGrid.fit(X,y)
    bestLogReg={'estimator' : logGrid.best_estimator_, 'info' : logGrid.cv_results_}

    with open(localDir + '/bestSpikeLogReg.pickle', 'wb') as f:
        pickle.dump(bestLogReg, f)
