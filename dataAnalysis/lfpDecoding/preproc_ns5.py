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
    ChannelData   = hf.getNSxData(datafile, elec_ids, start_time_s, data_time_s)
    simi_triggers = hf.getNSxData(datafile, 136, start_time_s, data_time_s)

    #
    if data_time_s == 'all':
        data_time_s = (simi_triggers['data'].index[-1] - simi_triggers['data'].index[0]) / (3e4)

    ch_idx  = chanToPlot
    hdr_idx = ChannelData['ExtendedHeaderIndices'][ChannelData['elec_ids'].index(chanToPlot)]
    electrodeLabel = ChannelData['extended_headers'][hdr_idx]['ElectrodeLabel']

    #pdb.set_trace()
    f,_ = hf.plotChan(ChannelData['data'], ChannelData['t'], chanToPlot,
        electrodeLabel = electrodeLabel, label = 'Raw data', mask = None,
        show = False, timeRange = (start_time_s, start_time_s + 30))

    origDataPath = fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_orig.npy'
    np.save(origDataPath, ChannelData['data'].values)

    #pdb.set_trace()
    if removeJumps:
        badData = {}
        # fill in overflow:
        #TODO: check if this expands into too much memory
        ChannelData['data'], overflowMask = hf.fillInOverflow(
            ChannelData['data'], fillMethod = 'average')
        badData.update({'overflow': overflowMask})

        # find unusual jumps in derivative or amplitude
        newBadData = hf.getBadContinuousMask(ChannelData['data'],
            ChannelData['samp_per_s'], ChannelData['t'],
            smoothing_ms = 0.5, nStdDiff = 50, nStdAmp = 100)
        badData.update(newBadData)
        # interpolate bad data
        for idx, row in ChannelData['data'].iteritems():
            #pdb.set_trace()
            mask = np.logical_or(badData['general'].to_dense(), badData['perChannelAmp'].loc[:,idx].to_dense())
            mask = np.logical_or(mask, badData['perChannelDer'].loc[:,idx].to_dense())
            ChannelData['data'].loc[:,idx] = hf.replaceBad(row, mask, typeOpt = 'interp')

        ChannelData['badData'] = badData
        # check interpolation results

        hf.plotChan(ChannelData['data'], ChannelData['t'], chanToPlot,
            electrodeLabel = electrodeLabel, label = 'Clean data',
            mask = [badData["general"].to_dense(), badData["perChannelAmp"].loc[:,ch_idx].to_dense(),
            badData["perChannelDer"].loc[:,ch_idx].to_dense(), badData["overflow"].loc[:,ch_idx].to_dense()],
            maskLabel = ["Flatline Dropout", "Amp Out of Bounds Dropout",
                "Derrivative Out of Bounds Dropout", "Overflow Dropout"],
            show = False, prevFig = f,
            timeRange = (start_time_s,start_time_s+30))

    if not os.path.exists(fileDir + '/dataAnalysisPreproc'):
        os.makedirs(fileDir + '/dataAnalysisPreproc')

    print('Saving clean data')

    dataPath = fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveClean.h5'
    ChannelData['data'].to_hdf(dataPath, 'data', mode = 'w')
    simi_triggers['data'].to_hdf(dataPath, 'simi', mode = 'a')
    ChannelData['t'] = pd.Series(ChannelData['t'], index = ChannelData['data'].index)
    ChannelData['t'].to_hdf(dataPath, 't', mode = 'a')

    metadata = {'channel':ChannelData, 'simiTrigger': simi_triggers}
    with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveCleanedMetadata.p', "wb" ) as f:
        pickle.dump(metadata, f, protocol=4 )

    print('Done cleaning data')

    plt.legend()
    plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] +
        '_ns5CleanFig.pdf')


    # # TODO: move this to its own function
    if printReport:
        print('Starting to write PDF Report.')
        origData = np.load(origDataPath, mmap_mode='r')
        origDataDF = pd.DataFrame(origData, index = ChannelData['data'].index, columns = ChannelData['data'].columns)
        pdfFile = fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_pdfReport.pdf'
        hf.pdfReport(ChannelData, origDataDF, badData = badData,
        pdfFilePath = pdfFile,
        spectrum = computeSpectrum, cleanSpectrum = None,
        origSpectrum = None, fr_start = fr_start, fr_stop = fr_stop)

    # get the spectrum TODO: not currently working
    if computeSpectrum:
        # spectrum function parameters
        R = 30 # target bandwidth for spectrogram
        clean_data_spectrum = hf.getSpectrogram(
            ChannelData['data'], ChannelData['elec_ids'],
            ChannelData['samp_per_s'], ChannelData['start_time_s'],
            ChannelData['t'], winLen_s, stepLen_s, R, fr_start = fr_start,
            fr_stop = fr_stop, whichChan = chanToPlot, plotting = False)
        plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] +
            '_SpectrumClean.png')
        plt.show()
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] +
            '_SpectrumClean.pickle', 'wb') as File:
                pickle.dump(plt.gcf(), File)
        spectrum_data = {'spectrum':clean_data_spectrum,
            'origin' : clean_data_spectrum['origin'],
            'winLen' : winLen_s, 'stepLen' : stepLen_s}
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_saveSpectrum.p', "wb" ) as f:
            pickle.dump(data, f, protocol=4 )
    else:
        clean_data_spectrum = {'origin': None}

    if compareBad and computeSpectrum:
        # get the spectrum
        origData = np.load(origDataPath, mmap_mode='r')
        origData_spectrum = hf.getSpectrogram(
            origData, ChannelData['elec_ids'], ChannelData['samp_per_s'], ChannelData['start_time_s'], ChannelData['t'], winLen_s, stepLen_s, R, fr_start = fr_start, fr_stop = fr_stop, whichChan = chanToPlot, plotting = False)

        plt.savefig(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.png')
        with open(fileDir + '/dataAnalysisPreproc' + fileName.split('.')[0] + '_SpectrumRaw.pickle', 'wb') as File:
            pickle.dump(plt.gcf(), File)

    pdb.set_trace()
    #x = input("Press any key")
