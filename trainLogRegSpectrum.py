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
ns5Name = '/saveSpectrumRightLabeled.p'
ns5File = localDir + ns5Name
ns5Data = pd.read_pickle(ns5File)

whichChans = range(10)
whichFreqs = ns5Data['channel']['spectrum']['fr'] < 30

spectrum = ns5Data['channel']['spectrum']['PSD']
t = ns5Data['channel']['spectrum']['t']
labels = ns5Data['channel']['spectrum']['LabelsNumeric']
y = labels

flatSpectrum = spectrum[whichChans, :, whichFreqs].transpose(1, 0, 2).to_frame().transpose()
X = flatSpectrum

skf = StratifiedKFold(n_splits=2, shuffle = True, random_state = 1)
logReg = LogisticRegression()

Cvalues=np.logspace(-1,3,2)

logGrid=GridSearchCV(logReg,{'penalty':['l1','l2']}, cv = skf, verbose = 3, n_jobs = -1)

if __name__ == '__main__':
    logGrid.fit(X,y)

    bestLogReg=logGrid.best_estimator_
    with open(localDir + '/bestSpectrumLogReg.pickle', 'wb') as f:
        pickle.dump({'estimator' : bestLogReg}, f)

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
