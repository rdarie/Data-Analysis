from dataAnalysis.helperFunctions.helper_functions import *
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np

parser = argparse.ArgumentParser()
#
parser.add_argument('--file', nargs='*', default = ['201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.nev','201612201054-Starbuck_Treadmill-Array1480_Right-Trial00002.nev'])
args = parser.parse_args()
argFile = args.file

dataName = ['/' + x.split('.')[0] + '_saveSpikeLabeled.p' for x in argFile]
whichChans = range(96)

#TODO: implement downsampling
skf = StratifiedKFold(n_splits=10, shuffle = True, random_state = 1)
LDA = LinearDiscriminantAnalysis()

solvers = ['svd']
shrinkages = ['auto']
componentCounts = [2]

parameters = {'solver': solvers, 'n_components' : componentCounts}
outputFileName = '/bestSpikeLDA_DownSampled.pickle'

trainSpikeMethod(dataName, whichChans, LDA, skf, parameters, outputFileName)
