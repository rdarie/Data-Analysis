import numpy as np

import pdb, re, warnings, string, os
from enum import Enum
from tkinter import *
from tkinter import filedialog

import matplotlib
from matplotlib import pyplot as plt

from dataAnalysis.helperFunctions.io_code import TDMS_to_dict

font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 16}

matplotlib.rc('font', **font)

class NORM(Enum):
    NONE = 0
    TOMAX = 1

gitPath = os.environ['GIT_BASE'] + '/Data-Analysis'

with open(gitPath + '/.serverHome') as f:
    serverHome = f.read().replace('\n', '')

def get_spikes(raw, chanLabels, filter_function, spike_dur, trig_sr,
    trig_idx, debugging):
    spike_num_samp = int(spike_dur*1e-3*trig_sr) #msec * 1e-3 msec/sec * samples/sec
    spike_time = np.arange(0, spike_dur, 1e3/round(trig_sr)) #step = 1e3 msec/sec / samples/sec
    spike_mat = np.zeros([len(trig_idx),spike_num_samp]) # empty array to hold spike_num_samp

    current_spike = 0

    for idx in trig_idx:
        spike = np.array(raw[chanLabels[0]]['data'][idx:idx+spike_num_samp]) #collect spike from raw data trace
        spike = filter_function(spike)
        spike_mat[current_spike,:] = spike
        current_spike += 1
        if debugging:
            plt.plot(spike_time,spike)

    mean_spike = np.mean(spike_mat, axis=0)
    std_spike = np.std(spike_mat,axis=0)
    return spike_mat, mean_spike, std_spike, spike_time

def get_trig_idx(raw):
    trig = raw['trigger']['data']
    trig_sr = raw['trigger']['sr'] # Samples/sec
    trig_diff = np.diff(trig) # triggers are square pulses, the first derivative of trig will have large peaks at the base of the slope
    trig_diff_bar = np.mean(trig_diff)
    trig_diff_std = np.std(trig_diff)
    trig_diff_threshold = trig_diff_bar + 30 * trig_diff_std
    first_pass_trig_idx = np.argwhere(trig_diff > trig_diff_threshold)

    trig_idx = [first_pass_trig_idx[0]]
    for idx in range(1,len(first_pass_trig_idx)):
        if first_pass_trig_idx[idx] > first_pass_trig_idx[idx-1] + 1:
            trig_idx.append(first_pass_trig_idx[idx])
    trig_idx = np.array(trig_idx)
    return trig_idx, trig, trig_sr, trig_diff_threshold, trig_diff

def plot_sta(whichChan, chanLabels, spike_dur, filter_function = lambda x: x,
    debugging = False, updateLog = True, override_ylim = 0,
    normalize = NORM.NONE):

    # get filename
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    TDMS_filename_all = filedialog.askopenfilenames(parent = root,
        title = 'Choose file(s)',
        initialdir = serverHome + '/Reversible lesions/Rodent Spinal Virus Transfection')
        # open window to get file name
    #pdb.set_trace()

    if not TDMS_filename_all: # if file not selected, select a default
        TDMS_filename_all = (serverHome +
            '/Reversible lesions/Rodent Spinal Virus Transfection/Survivor - 20160727/10_Survivor_HReflex_I1000_PW100us_A055.tdms',)
        warning_string = "No file selected, setting to default: %s" % TDMS_filename_all[0]
        warnings.warn(warning_string, UserWarning, stacklevel=2)

    filename_root = TDMS_filename_all[0].split(".");
    filename_path = '/'.join(filename_root[0].split('/')[:-1])+'/'

    if (not debugging) and (updateLog):
        notes_file = open(filename_path + 'notes.txt', 'w')

    for TDMS_filename in TDMS_filename_all:
        #
        raw = TDMS_to_dict(TDMS_filename, whichChan, chanLabels)

        # get a description of the what this trial contains, if appropriate
        print(TDMS_filename + '\n')
        if (not debugging) and updateLog:
            blurb = raw_input("What were the conditions during this recording?")
        else:
            blurb = "DEBUGGING"

        # get plot name based on file name
        filename_root = TDMS_filename.split(".")
        filename = filename_root[0].split('/')[-1]
        if not os.path.exists(filename_path + 'TRIG_figs/'):
            os.makedirs(filename_path + 'TRIG_figs/')
        plot_filename = filename_path + 'TRIG_figs/' + filename + "_TRIG.png"

        # if not debugging and updating the log, write the blurb to the log
        if (not debugging) and updateLog:
            notes_file.write(filename_root[0] + '\n\n')
            notes_file.write(blurb + '\n\n\n')

        trig_idx, trig, trig_sr, trig_diff_threshold, trig_diff = get_trig_idx(raw)

        fig = plt.figure(1, figsize=(10,5))
        fig_axes = fig.add_subplot(111)

        plt.plot(raw['trigger']['time'], trig, label='TTL trace (V)')
        plt.plot(raw['trigger']['time'][trig_idx], trig[trig_idx],'r*',
            label='Detected pulse start')
        plt.xlabel('Time (sec)')

        lgd = plt.legend(bbox_to_anchor=(0, 1.2, 1, 0.1),
            ncol=1, loc = 'center right', borderaxespad=0.)

        # Shrink current axis by 20%
        box = fig_axes.get_position()
        fig_axes.set_position([box.x0, box.y0, box.width, box.height * 0.8])

        #plt.title(blurb)

        if not debugging:
            plt.savefig(plot_filename)
            plt.clf()
        else:
            fig = plt.figure(2)
            fig_axes = fig.add_subplot(111)

        spike_mat, mean_spike, std_spike, spike_time = get_spikes(raw,
            chanLabels, filter_function, spike_dur, trig_sr, trig_idx, debugging)

        if normalize == NORM.TOMAX:
            norm_factor = mean_spike.max()
            spike_mat = spike_mat / norm_factor
            mean_spike = mean_spike / norm_factor
            std_spike = std_spike / norm_factor
        #pdb.set_trace()

        if debugging:
            plt.xlabel('Time (msec)')
            plt.ylabel(chanLabels[0] + ' (V)')
            if override_ylim:
                fig_axes.set_ylim(override_ylim)

            plt.figure(3)
            plt.plot(trig_diff, label='TTL trace first difference (V/timestep)')
            plt.plot(trig_idx, trig_diff[trig_idx],'r*',
                label='Detected pulse start')
            plt.axhline(y=trig_diff_threshold, xmin=0, xmax=1, linewidth=2,
                color = 'r', label = 'Detection Threshold')

        fig = plt.figure(4, figsize=(10,5))
        fig_axes = fig.add_subplot(111)
        plt.plot(spike_time, mean_spike,
            label = 'Mean ' + chanLabels[0] +' spike')
        plt.fill_between(spike_time, mean_spike+std_spike, mean_spike-std_spike,
            facecolor='blue', alpha = 0.3,
            label = 'Standard Deviation of '+ chanLabels[0] + ' spike')
        plt.xlabel('Time (msec)')
        plt.ylabel('Voltage (V)')
        if override_ylim:
            fig_axes.set_ylim(override_ylim)

        lgd = plt.legend(bbox_to_anchor=(0, 1.2, 1, 0.1),
           ncol=1, loc = 'center right', borderaxespad=0.)

        # Shrink current axis by 20%
        box = fig_axes.get_position()
        fig_axes.set_position([box.x0, box.y0, box.width, box.height * 0.8])

        #pdb.set_trace()

        if not os.path.exists(filename_path + chanLabels[0] +'_figs/'):
            os.makedirs(filename_path + chanLabels[0] +'_figs/')
        plot_filename = filename_path + chanLabels[0] +'_figs/'+ filename + "_STA_" + chanLabels[0] + ".png"
        if not debugging:
            plt.savefig(plot_filename, bbox_extra_artists=(lgd,))
            plt.clf()
        else:
            plt.show()

def compare_sta(whichChan, chanLabels, spike_dur, filter_function = lambda x: x,conditions = ['1', '2'], debugging = False, override_ylim = 0, normalize = NORM.NONE):

    # get color map
    use_color = plt.get_cmap('viridis')
    #pdb.set_trace()

    # get filename
    root = Tk()
    root.withdraw() # we don't want a full GUI, so keep the root window from appearing
    TDMS_filename_all = filedialog.askopenfilenames(title = 'Choose file(s)',
        initialdir = serverHome + '/Reversible lesions/Rodent Spinal Virus Transfection') # open window to get file name
    numFiles = len(TDMS_filename_all)
    #pdb.set_trace()

    if not TDMS_filename_all: # if file not selected, select a default
        TDMS_filename_all = (serverHome + '/Reversible lesions/Rodent Spinal Virus Transfection/Survivor - 20160727/10_Survivor_HReflex_I1000_PW100us_A055.tdms',)
        warning_string = "No file selected, setting to default: %s" % TDMS_filename_all[0]
        warnings.warn(warning_string, UserWarning, stacklevel=2)

    #string manipulation to get the name of the folder we're in, name of the file
    filename_root = TDMS_filename_all[0].split(".")
    filename_path = '/'.join(filename_root[0].split('/')[:-1])+'/'

    fig1 = plt.figure(1)
    fig1_axes = fig1.add_subplot(111)
    fig2 = plt.figure(2)
    fig2_axes = fig2.add_subplot(111)

    for counter,TDMS_filename in enumerate(TDMS_filename_all):
        raw = TDMS_to_dict(TDMS_filename,whichChan
        , chanLabels)

        trig_idx, trig, trig_sr, trig_diff_threshold, trig_diff = get_trig_idx(raw)

        filename_root = TDMS_filename.split(".");
        if debugging:
            plt.figure(1)

        spike_mat, mean_spike, std_spike, spike_time = get_spikes(raw, chanLabels, filter_function, spike_dur, trig_sr, trig_idx, debugging)
        #pdb.set_trace()
        if normalize == NORM.TOMAX:
            norm_factor = mean_spike.max()
            spike_mat = spike_mat / norm_factor
            mean_spike = mean_spike / norm_factor
            std_spike = std_spike / norm_factor

        if debugging:
            plt.xlabel('Time (msec)')
            plt.ylabel(chanLabels[0] + ' (V)')
            if override_ylim:
                fig1_axes.set_ylim(override_ylim)

        plt.figure(2)
        current_color = use_color(float(counter)/float(numFiles))[:3]
        #pdb.set_trace()
        plt.plot(spike_time,mean_spike, label = 'Mean ' + chanLabels[0] +', ' + conditions[counter], color = current_color, linewidth = 2)
        plt.fill_between(spike_time, mean_spike+std_spike, mean_spike-std_spike,facecolor=current_color, alpha = 0.3, label = 'Standard Deviation, ' + conditions[counter])
        plt.xlabel('Time (msec)')
        plt.ylabel('Voltage (V)')
        if override_ylim:
            fig2_axes.set_ylim(override_ylim)

        lgd = plt.legend(bbox_to_anchor=(0, 1.2, 1, 0.1),
           ncol=1, loc = 'center right', borderaxespad=0.)

        # Shrink current axis by 20%
        box = fig2_axes.get_position()
        fig2_axes.set_position([box.x0, box.y0, box.width, box.height * 0.8])

        plt.title('Signal Triggered Average Comparison')
        plot_filename = filename_root[0] + "_Compare_STA_" + chanLabels[0] + ".png"
    if not debugging:
        plt.savefig(plot_filename)
        plt.clf()
    else:
        plt.show()
