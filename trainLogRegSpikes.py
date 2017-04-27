from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os

from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.model_selection import validation_curve, GridSearchCV, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

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
    bestLogReg=logGrid.best_estimator_

    with open(localDir + '/bestSpikeLogReg.pickle', 'wb') as f:
        pickle.dump(bestLogReg, f)
        #plt.show(block = True)

"""
By default, the score computed at each CV iteration is the score method of
 the estimator. It is possible to change this by using the scoring parameter.

For LogReg, score is accuracy:

True positive (TP) = the number of cases correctly identified as patient

False positive (FP) = the number of cases incorrectly identified as patient

True negative (TN) = the number of cases correctly identified as healthy

False negative (FN) = the number of cases incorrectly identified as healthy

Accuracy: The accuracy of a test is its ability to differentiate the patient and healthy cases correctly. To estimate the accuracy of a test, we should calculate the proportion of true positive and true negative in all evaluated cases. Mathematically, this can be stated as:

Accuracy= (TP+TN) / (TP+TN+FP+FN)
"""
