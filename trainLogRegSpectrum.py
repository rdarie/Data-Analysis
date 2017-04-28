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

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
dataName = '/saveSpectrumRightLabeled.p'
dataFile = localDir + dataName
ns5Data = pd.read_pickle(dataFile)

whichChans = [1]
whichFreqs = ns5Data['channel']['spectrum']['fr'] < 10

spectrum = dataData['channel']['spectrum']['PSD']
#t = ns5Data['channel']['spectrum']['t']
labels = dataData['channel']['spectrum']['LabelsNumeric']
flatSpectrum = spectrum[whichChans, :, whichFreqs].transpose(1, 0, 2).to_frame().transpose()

X = flatSpectrum
y = labels

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = 1)
logReg = LogisticRegression(class_weight = 'balanced', max_iter = 500)

#cValues=np.logspace(-2,5,10)
cValues = [1]
solvers = ['liblinear']
penalties = ['l2']

logGrid=GridSearchCV(logReg, {'C': cValues,'penalty': penalties, 'solver' : solvers}, cv = skf, verbose = 4, scoring = 'f1_weighted', n_jobs = -1)

if __name__ == '__main__':
    logGrid.fit(X,y)

    bestLogReg={'estimator' : logGrid.best_estimator_, 'info' : logGrid.cv_results_}
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