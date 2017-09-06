import matplotlib, math
from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import pickle
from scipy.signal import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nspFile', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5')
parser.add_argument('--simiFile', default = 'Trial01_Step_Timing.txt')
args = parser.parse_args()
argNspFile = args.nspFile
argSimiFile = args.simiFile

font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 20
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rcParams.keys()

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

# Inits
localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']
fileName = '/' + argNspFile.split('.')[0] + '_saveSpectrum.p'
simiName = '/' + argSimiFile

dataFile = localDir + fileName
simiFile = localDir + simiName

# Read in simi text file
simiTable = pd.read_table(simiFile)
simiTable.drop(simiTable.index[[0]], inplace = True) # first rows contain giberrish
# Read in NSP data from preproc_ns5
data = pd.read_pickle(dataFile)

peakIdx, trigTimes = getCameraTriggers(data['simiTrigger'], plotting = True)
plt.show()

simiDf, gaitLabelFun, downLabelFun, upLabelFun = getGaitEvents(trigTimes, simiTable, ['ToeUp_Left Y', 'ToeDown_Left Y'],  plotting = True)

simiData = {'simiGait':simiDf, 'gaitLabelFun': gaitLabelFun, 'upLabelFun': upLabelFun, 'downLabelFun': downLabelFun}
pickle.dump(simiData, open( localDir + '/' + argSimiFile.split('.')[0] + '_saveSimi.p', "wb" ), protocol=4 )

#x = input("Press any key..")
