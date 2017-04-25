from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
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
ns5Name = '/saveRightLabeled.p'
ns5File = localDir + ns5Name
ns5Data = pd.read_pickle(ns5File)

whichChans = range(96)
whichFreqs = ns5Data['channel']['spectrum']['fr'] < 300
spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
labels = ns5Data['channel']['spectrum']['LabelsNumeric']
y = labels

flatSpectrum = spectrum[whichChans, :, whichFreqs].transpose(1, 0, 2).to_frame().transpose()
X = flatSpectrum

skf = StratifiedKFold(n_splits=3, shuffle = True, random_state = 1)
logReg = LogisticRegression()

Cvalues=np.logspace(-1,3,2)

logGrid=GridSearchCV(logReg,{'C': Cvalues,'penalty':['l1','l2']}, cv = skf, verbose = 2, n_jobs = -1)

if __name__ == '__main__':
    logGrid.fit(X,y)

    bestLogReg=logGrid.best_estimator_
    ylogreg=bestLogReg.predict(X)

    labelsNumeric = {'Neither': 0, 'Toe Up': 1, 'Toe Down': 2}
    numericLabels = {v: k for k, v in labelsNumeric.items()}
    predictedLabels = pd.Series([numericLabels[x] for x in ylogreg])

    plotting = True
    if plotting:
        #plot the spectrum
        upMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'] == 'Toe Up').values
        downMaskSpectrum = (ns5Data['channel']['spectrum']['Labels'] == 'Toe Down').values
        dummyVar = np.ones(ns5Data['channel']['spectrum']['t'].shape[0]) * 1

        upMaskSpectrumPredicted = (predictedLabels == 'Toe Up').values
        downMaskSpectrumPredicted = (predictedLabels == 'Toe Down').values

        fi = plotSpectrum(ns5Data['channel']['spectrum']['PSD'][1],
            ns5Data['channel']['samp_per_s'],
            ns5Data['channel']['start_time_s'],
            ns5Data['channel']['t'][-1],
            fr = ns5Data['channel']['spectrum']['fr'],
            t = ns5Data['channel']['spectrum']['t'],
            show = False)
        ax = fi.axes[0]
        ax.plot(ns5Data['channel']['spectrum']['t'][upMaskSpectrum], dummyVar[upMaskSpectrum], 'ro')
        ax.plot(ns5Data['channel']['spectrum']['t'][downMaskSpectrum], dummyVar[downMaskSpectrum] + 1, 'go')

        ax.plot(ns5Data['channel']['spectrum']['t'][upMaskSpectrumPredicted], dummyVar[upMaskSpectrumPredicted] + .5, 'mo')
        ax.plot(ns5Data['channel']['spectrum']['t'][downMaskSpectrumPredicted], dummyVar[downMaskSpectrumPredicted] + 1.5, 'co')

        with open(localDir + '/mySpectrumPlot.pickle', 'wb') as f:
            pickle.dump(fi, f)

        with open(localDir + '/bestSpectrumLogReg.pickle', 'wb') as f:
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
