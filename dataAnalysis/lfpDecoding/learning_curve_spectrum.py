from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle, os, sys, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default = 'bestSpectrumLDA_DownSampled.pickle')
parser.add_argument('--file', nargs='*', default =
    ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.ns5',
    '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5'])
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
modelFileName = '/' + argModel

estimator, estimatorInfo, whichChans, maxFreq = getEstimator(modelFileName)
modelName = getModelName(estimator)

ns5Names = ['/' + x.split('.')[0] + '_saveSpectrumLabeled.p' for x in argFile]

X, y, trueLabels = getSpectrumXY(ns5Names, whichChans, maxFreq)
ns5File = localDir + ns5Names[-1]
ns5Data = pd.read_pickle(ns5File)

origin = ns5Data['channel']['spectrum']['origin']
#spectrum = ns5Data['channel']['spectrum']['PSD']
#t = ns5Data['channel']['spectrum']['t']
#fr = ns5Data['channel']['spectrum']['fr']
#Fs = ns5Data['channel']['samp_per_s']
winLen = ns5Data['winLen']
stepLen = ns5Data['stepLen']
del(ns5Data)


suffix = modelName + '_winLen_' + str(winLen) + '_stepLen_' + str(stepLen) + '_from_' + origin


skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
title = 'Learning curve for ' + suffix
if __name__ == '__main__':
    fiLc = plotLearningCurve(estimator, title, X, y, ylim=None, cv=skf,
        scoreFun = ROCAUC_ScoreFunction)

plt.savefig(localDir + '/' + modelName + '/spectrum_LearningCurve_'+ suffix + '.png')
with open(localDir + '/' + modelName + '/spectrum_LearningCurve_'+ suffix + '.pickle', 'wb') as f:
    pickle.dump(fiLc, f)
