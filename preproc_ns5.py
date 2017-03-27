# -*- coding: utf-8 -*-
"""
Example of how to extract and plot continuous data saved in Blackrock nsX data files
current version: 1.1.1 --- 07/22/2016

@author: Mitch Frankel - Blackrock Microsystems
"""

"""
Version History:
v1.0.0 - 07/05/2016 - initial release - requires brpylib v1.0.0 or higher
v1.1.0 - 07/12/2016 - addition of version checking for brpylib starting with v1.2.0
                      minor code cleanup for readability
v1.1.1 - 07/22/2016 - now uses 'samp_per_sec' as returned by NsxFile.getdata()
                      minor modifications to use close() functionality of NsxFile class
"""
import matplotlib, math
from helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import libtfr
import sys
import pickle

font_opts = {'family' : 'arial',
        'weight' : 'bold',
        'size'   : 30
        }

fig_opts = {
    'figsize' : (10,5),
    }

matplotlib.rcParams.keys()

matplotlib.rc('font', **font_opts)
matplotlib.rc('figure', **fig_opts)

# Inits
fileDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings/201612201054-Starbuck_Treadmill/Right_Array/';
fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5';

datafile = fileDir + fileName

elec_ids     = range(1,97)              # 'all' is default for all (1-indexed)
start_time_s = 0                        # 0 is default for all
data_time_s  = 'all'                    # 'all' is default for all
whichChan    = 2                        # 1-indexed

simi_triggers, _, _ = getNSxData(datafile, 136, start_time_s, data_time_s)

cont_data, _, extended_headers = getNSxData(datafile, elec_ids, start_time_s, data_time_s)
badData = getBadDataMask(cont_data, extended_headers, plotting = False, smoothing_ms = 1)

pdfFile = fileDir + 'Python/pdfReport.pdf'
pdfReport(cont_data, extended_headers, mask = badData, pdfFilePath = pdfFile)

f,_ = plot_chan(cont_data, extended_headers, whichChan, mask = None, show = False)
# interpolate bad data
cont_data['data'].apply(replaceBad, raw = False, args = (badData, 'interp'))
# check interpolation results
plot_chan(cont_data, extended_headers, whichChan, mask = badData, show = True, prevFig = f)

# spectrum function parameters
winLen_s = 0.1
stepLen_fr = 0.25 # window step as a fraction of window length
R = 50 # target bandwidth for spectrogram

spectrum = get_spectrogram(cont_data, winLen_s, stepLen_fr, R, whichChan)

data = {'channel':cont_data, 'headers':extended_headers, 'spectrum':spectrum, 'simiTrigger': simi_triggers}
pickle.dump(data, open( fileDir + "Python/save.p", "wb" ), protocol=4 )
