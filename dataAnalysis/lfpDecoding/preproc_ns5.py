# -*- coding: utf-8 -*-
"""
Based on example of how to extract and plot continuous data saved in Blackrock nsX data files
from brpy version: 1.1.1 --- 07/22/2016

@author: Radu Darie
"""

import matplotlib, math, pdb
import dataAnalysis.helperFunctions.helper_functions as hf
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
def preproc_ns5(stepLen_s = 0.05, winLen_s = 0.1, fr_start = 5, fr_stop = 1000,\
 elec_ids = range(1, 97),  chanToPlot = 90,\
   fileName = '201612201054-Starbuck_Treadmill-Array1480_Right-Trial00001.ns5',\
    fileDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Training/Flywheel Logs/Murdoc',\
     start_time_s = 0, data_time_s = 'all', compareBad = False,\
      fillOverflow = False, removeJumps = True, printReport = False,\
       computeSpectrum = True, saveNSx = None):

    print("Preprocessing spectral data with a window length of {:4.4f} seconds and a step length of {:4.4f} seconds".format(winLen_s, stepLen_s))

    # Reformat figures
    font_opts = {'family' : 'arial',
            'weight' : 'bold',
            'size'   : 20
            }
    fig_opts = {
        'figsize' : (10,5),
        }
    legend_opts = {
        'fontsize' : 8,
        }

    matplotlib.rc('font', **font_opts)
    matplotlib.rc('figure', **fig_opts)
    matplotlib.rc('legend', **legend_opts)

    # Inits
    fileName = '/' + fileName
    datafile = fileDir + fileName
    #pdb.set_trace()
    simi_triggers = hf.getNSxData(datafile, 136, start_time_s, data_time_s)
    ChannelData   = hf.getNSxData(datafile, elec_ids, start_time_s, data_time_s, memMapFile = True)

    #
    if data_time_s == 'all':
        data_time_s = simi_triggers['data'].index[-1] / (3e4)

    ch_idx  = chanToPlot
    hdr_idx = ChannelData['ExtendedHeaderIndices'][ChannelData['elec_ids'].index(chanToPlot)]
    electrodeLabel = ChannelData['extended_headers'][hdr_idx]['ElectrodeLabel']
    #pdb.set_trace()

    f,_ = hf.plotChan(ChannelData['data'], ChannelData['t'], chanToPlot, electrodeLabel = electrodeLabel, label = 'Raw data', mask = None, show = False, timeRange = (start_time_s,start_time_s+30))

    clean_data = deepcopy(ChannelData['data'])
    # fill in overflow:
    if fillOverflow:
        clean_data, whereOverflow = hf.fillInOverflow(clean_data, fillMethod = 'average')
    #pdb.set_trace()
    if removeJumps:
        badData = hf.getBadContinuousMask(clean_data, ChannelData['samp_per_s'], ChannelData['t'], smoothing_ms = 0.5, nStdDiff = 50, nStdAmp = 100)
        # interpolate bad data
        for idx, row in clean_data.iteritems():
            mask = np.logical_or(badData['general'], badData[idx]['perChannelAmp'])
            mask = np.logical_or(mask, badData[idx]['perChannelDer'])
            row = hf.replaceBad(row, mask, typeOpt = 'interp')

        ChannelData['badData'] = badData
        # check interpolation results
        #pdb.set_trace()
        dip_mask = np.full(len(clean_data.index), False, dtype = np.bool)
        dip_mask[whereOverflow[ch_idx]['dips']] = True
        return_mask = np.full(len(clean_data.index), False, dtype = np.bool)
        return_mask[whereOverflow[ch_idx]['returns']] = True
        #
        #
        hf.plotChan(clean_data, ChannelData['t'], chanToPlot, electrodeLabel = electrodeLabel, label = 'Clean data', mask = [badData["general"], badData[ch_idx]["perChannelAmp"], badData[ch_idx]["perChannelDer"], dip_mask, return_mask],
            maskLabel = ["Flatline Dropout", "Amp Out of Bounds Dropout", "Derrivative Out of Bounds Dropout", "Overflow Dips", "Overflow Returns"], show = False, prevFig = f, timeRange = (start_time_s,start_time_s+30))

    if not os.path.exists(fileDir + '/dataAnalysisPreproc'):
        os.makedirs(fileDir + '/dataAnalysisPreproc')

    plt.legend()
    #plt.show()
    #pdb.set_trace()
    plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_ns5Clean.png')
    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_ns5Clean.pickle', 'wb') as File:
        pickle.dump(f, File)

    # spectrum function parameters
    R = 30 # target bandwidth for spectrogram

    # get the spectrum
    if computeSpectrum:
        clean_data_spectrum = hf.getSpectrogram(
            clean_data, ChannelData['elec_ids'], ChannelData['samp_per_s'], ChannelData['start_time_s'], ChannelData['t'], winLen_s, stepLen_s, R, fr_start = fr_start, fr_stop = fr_stop, whichChan = chanToPlot, plotting = False)
        plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumClean.png')
        plt.show()
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumClean.pickle', 'wb') as File:
            pickle.dump(plt.gcf(), File)
    else:
        clean_data_spectrum = {'origin': None}

    if compareBad and computeSpectrum:
        # get the spectrum
        ChannelData['spectrum'] = hf.getSpectrogram(
            ChannelData['data'], ChannelData['elec_ids'], ChannelData['samp_per_s'], ChannelData['start_time_s'], ChannelData['t'], winLen_s, stepLen_s, R, fr_start = fr_start, fr_stop = fr_stop, whichChan = chanToPlot, plotting = False)

        plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.png')
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.pickle', 'wb') as File:
            pickle.dump(plt.gcf(), File)

    if printReport:
        print('Starting to write PDF Report.')
        pdfFile = fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_pdfReport.pdf'
        hf.pdfReport(ChannelData, clean_data, badData = badData, whereOverflow = whereOverflow, pdfFilePath = pdfFile, spectrum = computeSpectrum, clean_data_spectrum = clean_data_spectrum, fr_start = fr_start, fr_stop = fr_stop)

    ChannelData['data'] = clean_data
    data = {'channel':clean_data, 'simiTrigger': simi_triggers}
    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveCleaned.p', "wb" ) as f:
        pickle.dump(data, f, protocol=4 )

    if computeSpectrum:
        spectrum_data = {'spectrum':clean_data_spectrum,
            'origin' : clean_data_spectrum['origin'],
            'winLen' : winLen_s, 'stepLen' : stepLen_s}
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
            pickle.dump(data, f, protocol=4 )

    #x = input("Press any key")
