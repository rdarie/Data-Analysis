from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpikeLDA.pickle')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev'])
args = parser.parse_args()
argModel = args.model
argFile = args.file

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
dataNames = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
modelFileName = '/' + argModel

estimator, estimatorInfo, _, _ = getEstimator(modelFileName)
modelName = getModelName(estimator)
suffix = modelName

whichChans = list(range(96))
X, y, trueLabels = getSpikeXY(dataNames, whichChans)

skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
title = 'Learning curve for ' + suffix
if __name__ == '__main__':
    fiLc = plotLearningCurve(estimator, title, X, y, ylim=None, cv=skf,
        n_jobs=-1, pre_dispatch = 'n_jobs', scoreFun = ROCAUC_ScoreFunction)

plt.savefig(localDir + '/' + modelName + '/spike_LearningCurve_'+ suffix + '.png')
with open(localDir + '/' + modelName + '/spike_LearningCurve_'+ suffix + '.pickle', 'wb') as f:
    pickle.dump(fiLc, f)
