from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

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

localDir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'
spikeName = 'saveSpikeRightLabeled.p'
spikeFile = localDir + spikeName
spikeData = pd.read_pickle(spikeFile)
spikes = spikeData['spikes']
binCenters = spikeData['binCenters']
spikeMat = spikeData['spikeMat']
binWidth = spikeData['binWidth']

chans = spikeMat.columns.values[np.array([not isinstance(x, str) for x in spikeMat.columns.values], dtype = bool)]

X = spikeMat[chans]
y = spikeMat['LabelsNumeric']
skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1)
logReg = LogisticRegression()

Cvalues=np.logspace(-1,4,3)

logGrid=GridSearchCV(logReg,{'C': Cvalues,'penalty':['l1','l2']}, cv = skf, n_jobs = -1, verbose = 3)
logGrid.fit(X,y)
bestLogReg=logGrid.best_estimator_
ylogreg=cross_val_predict(bestLogReg,X,y)

labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
numericLabels = {v: k for k, v in labelsNumeric.items()}
predictedLabels = pd.Series([numericLabels[x] for x in ylogreg])

plotting = True
if plotting:
    #Plot the spikes
    fi = plotBinnedSpikes(X, binCenters, chans, show = False)

    upMaskSpikes = (spikeMat['Labels'] == 'Toe Up').values
    downMaskSpikes = (spikeMat['Labels'] == 'Toe Down').values

    upMaskSpikesPredicted = (predictedLabels == 'Toe Up').values
    downMaskSpikesPredicted = (predictedLabels == 'Toe Down').values

    dummyVar = np.ones(binCenters.shape[0]) * 1
    ax = fi.axes[0]
    ax.plot(binCenters[upMaskSpikes], dummyVar[upMaskSpikes], 'ro')
    ax.plot(binCenters[downMaskSpikes], dummyVar[downMaskSpikes] + 1, 'go')

    ax.plot(binCenters[upMaskSpikesPredicted], dummyVar[upMaskSpikesPredicted] + .5, 'mo')
    ax.plot(binCenters[downMaskSpikesPredicted], dummyVar[downMaskSpikesPredicted] + 1.5, 'co')

    with open(localDir + 'myplot.pickle', 'wb') as f:
        pickle.dump(fi, f)
    plt.show(block = True)

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
