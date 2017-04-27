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

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1)
logReg = LogisticRegression()

Cvalues=np.logspace(-1,4,3)

logGrid=GridSearchCV(logReg,{'C': Cvalues,'penalty':['l1','l2']}, cv = skf, n_jobs = -1, verbose = 3)

if __name__ == '__main__':
    logGrid.fit(X,y)
    bestLogReg={'estimator' : logGrid.best_estimator_}
    pdb.set_trace()

    with open(localDir + '/bestSpikeLogReg.pickle', 'wb') as f:
        pickle.dump(bestLogReg, f)
