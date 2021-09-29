import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from scipy.stats import median_abs_deviation
import pdb
import matplotlib.pyplot as plt
"""
based original work by M. Trofficus
at https://github.com/MichaelisTrofficus/hampel_filter
"""


def hampel(
        ts, window_size=5, n=3, imputation=False,
        debug_plots=False
        ):
    """
    Median absolute deviation (MAD) outlier in Time Series
    :param ts: a pandas Series object representing the timeseries
    :param window_size: total window size will be computed as 2*window_size + 1
    :param n: threshold, default is 3 (Pearson's rule)
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
    if type(n) != int:
        raise ValueError("Window size must be of type integer.")
    else:
        if n < 0:
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
    #  #############################
    #  if True:
    #      ts = ts.loc[:, ['seg0_utah80']]
    #      ts_cleaned = ts_cleaned.loc[:, ['seg0_utah80']]
    #  #############################
    #
    rolling_median = pd.DataFrame(
        median_filter(ts_cleaned.to_numpy(), size=(window_size*2+1, 1)),
        index=ts_cleaned.index, columns=ts_cleaned.columns)
    rolling_sigma = k * pd.DataFrame(
        median_filter(
            (ts_cleaned - rolling_median).abs().to_numpy(),
            size=(window_size*2+1, 1)),
        index=ts_cleaned.index, columns=ts_cleaned.columns)
    '''rolling_sigma = (
        k *
        (
            ts_cleaned
            .rolling(window_size*2, center=True, axis='index')
            .apply(median_abs_deviation, raw=True)
            .fillna(method='bfill')
            .fillna(method='ffill'))
        )'''
    #
    outlier_mask = np.abs(ts_cleaned - rolling_median) >= (n * rolling_sigma)
    #
    if imputation:
        for cN in ts_cleaned.columns:
            ts_cleaned.loc[outlier_mask[cN], cN] = rolling_median.loc[outlier_mask[cN], cN]
        #
        if debug_plots:
            fig, ax = plt.subplots()
            twinAx = ax.twinx()
            t = ts_cleaned.index.to_numpy()
            raw = ts.iloc[:, 0].to_numpy()
            cleaned = ts_cleaned.iloc[:, 0].to_numpy()
            mad = rolling_sigma.iloc[:, 0].to_numpy()
            ax.plot(t, raw, label='raw')
            ax.plot(t, cleaned, label='cleaned')
            ax.legend(loc='lower left')
            twinAx.plot(t, mad, label='mad')
            twinAx.set_xlabel('mad')
            plt.show()
        if type(ts) == pd.Series:
            return ts_cleaned['data']
        else:
            return ts_cleaned
    #
    return outlier_mask