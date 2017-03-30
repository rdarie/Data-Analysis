import matplotlib, math
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import libtfr
import sys
import pickle
from scipy.signal import *

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
fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/';
fileName = 'Python/save.p'
simiName = 'Trial01_Step_Timing.txt'

dataFile = fileDir + fileName
simiFile = fileDir + simiName

# Read in simi text file
simiTable = pd.read_table(simiFile)
simiTable.drop(simiTable.index[[0,1,2]], inplace = True) # first rows contain giberrish
# Read in NSP data from preproc_ns5
data = pd.read_pickle(dataFile)

peakIdx, trigTimes = getCameraTriggers(data['simiTrigger'], plotting = True)

simiDf, gaitLabelFun, downLabelFun, upLabelFun = getGaitEvents(trigTimes, simiTable, plotting = True)

simiData = {'simiGait':simiDf, 'gaitLabelFun': gaitLabelFun, 'upLabelFun': upLabelFun, 'downLabelFun': downLabelFun}
pickle.dump(simiData, open( fileDir + "Python/saveSimi.p", "wb" ), protocol=4 )
