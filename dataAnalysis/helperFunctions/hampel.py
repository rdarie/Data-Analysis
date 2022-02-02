import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
import pdb
import matplotlib.pyplot as plt
import seaborn as sns
from astropy.convolution import convolve, interpolate_replace_nans
"""
based original work by M. Trofficus
at https://github.com/MichaelisTrofficus/hampel_filter
"""


def hampel(
        ts, window_size=5, thresh=3, imputation=False,
        average_across_channels=False,
        debug_plots=False
        ):
    """
    Median absolute deviation (MAD) outlier in Time Series
    :param ts: a pandas Series or DataFrame object representing the timeseries
    :param window_size: total window size
    :param thresh: threshold, default is 3 (Pearson's rule)
    :param imputation: If set to False, then the algorithm will be used for outlier detection.
        If set to True, then the algorithm will also imput the outliers with the rolling median.
    :return: Returns a boolean mask indicating outlier samples if imputation=False and the corrected timeseries if imputation=True
    """
    #
    if not ((type(ts) == pd.Series) or (type(ts) == pd.DataFrame)):
        raise ValueError("Timeserie object must be of type pandas.Series or pandas.DataFrame.")
    #
    if type(window_size) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if window_size <= 0:
            raise ValueError("Window size must be more than 0.")
    #
    if thresh < 0:
        raise ValueError("Window size must be equal or more than 0.")
    #
    # Copy the DataFrame object. This will be the cleaned timeseries
    if type(ts) == pd.Series:
        ts_cleaned = ts.copy().to_frame(name='data')
    else:
        ts_cleaned = ts.copy()
    # Constant scale factor, which depends on the distribution
    # In this case, we assume normal distribution
    k = 1.4826
    #
    #############################
    ## DEBUGGING ONLY!
    # 
    # if True:
    #     selectColumns = ['seg0_utah78', 'seg0_utah88', 'seg0_utah68', 'seg0_utah58', 'seg0_utah48', 'seg0_utah1']
    #     ts = ts.loc[:, selectColumns]
    #     ts_cleaned = ts_cleaned.loc[:, selectColumns]
    #############################
    #
    rolling_median = pd.DataFrame(
        median_filter(ts_cleaned.to_numpy(), size=(window_size, 1)),
        index=ts_cleaned.index, columns=ts_cleaned.columns)
    absolute_deviation = (ts_cleaned - rolling_median).abs()
    rolling_sigma = k * pd.DataFrame(
        median_filter(
            absolute_deviation.to_numpy(),
            size=(window_size, 1)),
        index=ts_cleaned.index, columns=ts_cleaned.columns)
    #
    scaled_absolute_deviation = (absolute_deviation / rolling_sigma)
    # fill zeros with 0
    scaled_absolute_deviation.mask(absolute_deviation == 0, 0, inplace=True)
    #
    posInfMask = np.isposinf(scaled_absolute_deviation)
    if posInfMask.any().any():
        print(
            '{} of {} samples yield positive infinite scaled absolute deviation ({:0.3f} pct.)'
            .format(posInfMask.sum().sum(), posInfMask.size, 100 * posInfMask.sum().sum() / posInfMask.size))
        nonInfMax = scaled_absolute_deviation.mask(posInfMask, other=-1e3).max().max()
        scaled_absolute_deviation.mask(posInfMask, other=0., inplace=True)
    #
    negInfMask = np.isneginf(scaled_absolute_deviation)
    if negInfMask.any().any():
        print(
            '{} of {} samples yield negative infinite scaled absolute deviation ({:0.3f} pct.)'
            .format(negInfMask.sum().sum(), negInfMask.size, 100 * negInfMask.sum().sum() / negInfMask.size))
        nonInfMin = scaled_absolute_deviation.mask(negInfMask, other=1e3).min().min()
        scaled_absolute_deviation.mask(negInfMask, other=0., inplace=True)
    nanMask = scaled_absolute_deviation.isna()
    if nanMask.any().any():
        print(
            '{} of {} samples yield nan scaled absolute deviation ({:0.3f} pct.)'
            .format(nanMask.sum().sum(), nanMask.size, 100 * nanMask.sum().sum() / nanMask.size))
        scaled_absolute_deviation.fillna(n+1, inplace=True)
    del absolute_deviation
    #
    if average_across_channels:
        outlier_mask_across = (scaled_absolute_deviation.mean(axis='columns') >= thresh)
        outlier_proportion = outlier_mask_across.sum() / outlier_mask_across.shape[0]
        outlier_mask = pd.DataFrame(
            np.tile(outlier_mask_across, (scaled_absolute_deviation.shape[1], 1)).T,
            index=scaled_absolute_deviation.index,
            columns=scaled_absolute_deviation.columns)
    else:
        outlier_mask = (scaled_absolute_deviation >= thresh)
        outlier_proportion = outlier_mask.sum().sum() / outlier_mask.size
    ####
    outlier_mask = ((
        outlier_mask | outlier_mask.shift(1).fillna(False)
        ) | outlier_mask.shift(-1).fillna(False))
    ####
    proportion_message = (
        '{:.1f} pct. of samples cross the threshold ({})'.format(
            100 * outlier_proportion, thresh)
        )
    print(proportion_message)
    if debug_plots:
        fig, ax = plt.subplots(3, 1)
    else:
        fig, ax = None, None
    if imputation:
        '''
        pre_clean = pd.DataFrame(
            interpolate_replace_nans(
                ts_cleaned.mask(outlier_mask, other=np.nan).to_numpy(),
                kernel=np.ones((window_size, 1))),
            index=ts_cleaned.index, columns=ts_cleaned.columns).fillna('pad')
        filler = pd.DataFrame(
            median_filter(
                pre_clean.to_numpy(), size=(window_size, 1)),
            index=ts_cleaned.index, columns=ts_cleaned.columns)
        ts_cleaned.mask(outlier_mask, other=filler, inplace=True)
        '''
        ts_cleaned.mask(outlier_mask, other=rolling_median, inplace=True)
        #
        if debug_plots:
            t = ts_cleaned.index.to_numpy()
            palette1=['b', 'g', 'r']
            palette2=['c', 'y', 'm']
            for cIdx, cN in enumerate(['seg0_utah78', 'seg0_utah1']):
                raw = ts.loc[:, cN].to_numpy()
                cleaned = ts_cleaned.loc[:, cN].to_numpy()
                ax[0].plot(t, raw, ls='--', lw=1., c=palette1[cIdx], alpha=0.5, label='{} (raw)'.format(cN))
                ax[0].plot(t, cleaned, lw=0.5, c=palette2[cIdx], label='{} (cleaned)'.format(cN))
            ax[0].set_ylabel('LFP (uV)')
            ax[0].legend(loc='lower left')
            #
            if average_across_channels:
                sad = scaled_absolute_deviation.mean(axis='columns').to_numpy()
                ax[1].plot(t, sad, c='k', label='(scaled abs. dev.)')
                sns.ecdfplot(sad, ax=ax[2], c='k')
            else:
                for cIdx, cN in enumerate(['seg0_utah78', 'seg0_utah1']):
                    sad = scaled_absolute_deviation.loc[:, cN].to_numpy()
                    ax[1].plot(t, sad, c=palette1[cIdx], label='{} (scaled abs. dev.)'.format(cN))
                sns.ecdfplot(scaled_absolute_deviation.to_numpy().flatten(), ax=ax[2], c='k')
            ax[1].axhline(thresh, c='r', label='threshold')
            ax[1].set_ylabel('Scaled abs. dev. (a.u.)')
            ax[1].set_xlabel('Time (sec)')
            ax[1].legend(loc='upper left')
            ax[1].get_shared_x_axes().join(ax[1], ax[0])
            #
            ax[2].set_xlim([0, 2 * thresh])
            ax[2].set_xlabel('Scaled abs. dev.')
            ax[2].set_ylabel('Cummulative proportion')
            ax[2].axvline(thresh, c='r', label='threshold')
            ax[2].legend(loc='lower right')
            #
            fig.suptitle(proportion_message)
        if average_across_channels:
            scaled_absolute_deviation = scaled_absolute_deviation.mean(axis='columns')
        if type(ts) == pd.Series:
            return ts_cleaned['data'], scaled_absolute_deviation, (fig, ax)
        else:
            return ts_cleaned, scaled_absolute_deviation, (fig, ax)
    #
    if average_across_channels:
        scaled_absolute_deviation = scaled_absolute_deviation.mean(axis='columns')
    return outlier_mask, scaled_absolute_deviation, (fig, ax)

def defaultSpiketrainHampel(waveDF, spkTrain):
    print('running defaultSpiketrainHampel!!!!')
    res, _, _ =  hampel(waveDF, average_across_channels=True, window_size=31, thresh=3, imputation=True)
    return res