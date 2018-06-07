# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""

import matplotlib, math
from dataAnalysis.helperFunctions.helper_functions import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys, os
import pickle
from copy import *
import argparse

"""
parser = argparse.ArgumentParser()
parser.add_argument('--stepLen', default = '0.05')
parser.add_argument('--winLen', default = '0.1')
parser.add_argument('--file', default = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5')
parser.add_argument('--folder', default = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc')

args = parser.parse_args()
argWinLen = float(args.winLen)
argStepLen = float(args.stepLen)
argFile = args.file
fileDir = args.folder
"""
def preproc_ns5(stepLen_s = 0.05, winLen_s = 0.1, fr_start = 5, fr_stop = 1000, elec_ids = range(1, 97),  chanToPlot = 90,  fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5', fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc', start_time_s = 0, data_time_s = 60, compareBad = False, printReport = False):

    print("Preprocessing spectral data with a window length of {:4.4f} seconds and a step length of {:4.4f} seconds".format(winLen_s, stepLen_s))

    # Reformat figures
    font_opts = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 20
            }
    fig_opts = {
        'figsize' : (10,5),
        }

    matplotlib.rc('font', **font_opts)
    matplotlib.rc('figure', **fig_opts)

    # Inits
    fileName = '/' + fileName
    datafile = fileDir + fileName
    simi_triggers = getNSxData(datafile, 136, start_time_s, data_time_s)

    ChannelData = getNSxData(datafile, elec_ids, start_time_s, data_time_s)

    ch_idx  = ChannelData['elec_ids'].index(chanToPlot)

    badData = getBadContinuousMask(ChannelData, plotting = chanToPlot, smoothing_ms = 0.5)

    f,_ = plotChan(ChannelData, chanToPlot, label = 'Raw data', mask = None, show = False)

    clean_data = deepcopy(ChannelData)

    # interpolate bad data
    for idx, row in clean_data['data'].iteritems():
        mask = np.logical_or(badData['general'], badData['perChannel'][idx])
        row = replaceBad(row, mask, typeOpt = 'interp')

    clean_data['badData'] = badData
    # check interpolation results
    plot_mask = np.logical_or(badData['general'], badData['perChannel'][ch_idx])
    plotChan(clean_data, chanToPlot, label = 'Clean data', mask = plot_mask,
        maskLabel = "Dropout", show = True, prevFig = f)

    if not os.path.exists(fileDir + '/dataAnalysisPreproc'):
        os.makedirs(fileDir + '/dataAnalysisPreproc')

    plt.legend()
    plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_ns5Clean.png')
    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_ns5Clean.pickle', 'wb') as File:
        pickle.dump(f, File)
    # spectrum function parameters
    R = 30 # target bandwidth for spectrogram

    # get the spectrum
    clean_data['spectrum'] = getSpectrogram(
        clean_data, winLen_s, stepLen_s, R, fr_start = fr_start, fr_stop = fr_stop, whichChan = chanToPlot, plotting = True)

    plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumClean.png')
    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumClean.pickle', 'wb') as File:
        pickle.dump(plt.gcf(), File)

    if compareBad:
        # get the spectrum
        ChannelData['spectrum'] = getSpectrogram(
            ChannelData, winLen_s, stepLen_s, R, fr_start = fr_start, fr_stop = fr_stop, whichChan = chanToPlot, plotting = True)

        plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.png')
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.pickle', 'wb') as File:
            pickle.dump(plt.gcf(), File)


    data = {'channel':clean_data, 'simiTrigger': simi_triggers,
        'origin' : clean_data['spectrum']['origin'],
        'winLen' : winLen_s, 'stepLen' : stepLen_s}

    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
        pickle.dump(data, f, protocol=4 )

    if printReport:
        print('Starting to write PDF Report.')

        pdfFile = fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_pdfReport.pdf'
        pdfReport(ChannelData, clean_data, badData = badData, pdfFilePath = pdfFile, spectrum = True, fr_start = fr_start, fr_stop = fr_stop)

    #x = input("Press any key")
