import numpy as np
import pywt
import matplotlib.pyplot as plt
import pdb
from dataAnalysis.helperFunctions.pywt_helpers import *
import seaborn as sns
sns.set()
sns.set_color_codes("dark")
sns.set_context("notebook")
sns.set_style("darkgrid")

'''hBound = 4
lBound = 2
bandwidth = (hBound - lBound) / 2  # Hz
center = (hBound + lBound) / 2  # Hz'''
bandwidth = .5
center = 3
dt = 1e-3
# dt = 2e-4
targetScale = 50
maxScale = 100
sigma = morletBToSigma(bandwidth, fs=dt ** -1, scale=targetScale)
B, C = freqsToMorletParams(bandwidth, center, fs=dt ** -1, scale=targetScale)
print('Target center = {:.3f} Hz, bandwidth = {:.3f} Hz'.format(center, bandwidth))
waveletName = 'cmor{:.3f}-{:.3f}'.format(B, C)
print('wavelet: {}'.format(waveletName))
targetWidth = 2 * np.ceil((6 * maxScale * np.sqrt(B / 2) - 1) / maxScale)
bws, fcs, bw_ratios = plotKernels(
    pywt.ContinuousWavelet(waveletName),
    np.asarray([1, 20, targetScale]),
    dt=dt, precision=16, verbose=True,
    width=targetWidth)

thisWav = pywt.ContinuousWavelet('cmor2.0-1.0')
theseScales = np.asarray([5, 10, 20, 30])
#theseScales = np.linspace(5, 20, 10)
theseBws, theseFCs, theseBwrs = plotKernels(thisWav, theseScales, dt=1e-3)
pdb.set_trace()
bwDict = {}
for thisB in np.arange(2, 10):
    temp = {}
    for thisC in np.arange(2, 5):
        bws, fcs, bw_ratios = plotKernels(
            pywt.ContinuousWavelet('cmor{:.1f}-{:.1f}'.format(thisB, thisC)),
            theseScales, dt=dt)
        temp[thisC] = pd.DataFrame({'bandwidth': bws, 'bw_ratio': bw_ratios}, index=theseScales)
        temp[thisC].index.name = 'scale'
        plt.show(block=False)
    bwDict[thisB] = pd.concat(temp, names='C')
plt.close('all')
bwDF = pd.concat(bwDict, names='B').reset_index()
bwDF.loc[:, 'B'] = bwDF['B'].astype(float) # units of samples ** 2?
bwDF.loc[:, 'sigma'] = (bwDF['B'] / 2).apply(np.sqrt) # units of samples
bwDF.loc[:, 'sigma_adj'] = bwDF['sigma'] * dt
#
bwDF.loc[:, 'scaledbw'] = bwDF['bandwidth'].multiply(bwDF['scale']) # units of Hz
bwDF.loc[:, 'scaledvariance'] = bwDF['scaledbw'] ** 2
bwDF.loc[:, 'sqrtB'] = bwDF['B'].apply(np.sqrt)
# bwDF.loc[:, 'gamma'] = (np.pi * (bwDF['B'] * dt * 4).apply(np.sqrt)) ** (-1)
bwDF.loc[:, 'gamma'] = (2 * dt * np.pi * bwDF['scale'] * bwDF['B'].apply(np.sqrt)) ** (-1)
bwDF.loc[:, 'gammaratio'] = bwDF['bandwidth'] / bwDF['gamma']

fig, ax = plt.subplots()
sns.lineplot(x='gamma', y='bandwidth', data=bwDF, ax=ax)
fig, ax = plt.subplots()
sns.violinplot(x='C', y='gammaratio', data=bwDF, ax=ax)
fig, ax = plt.subplots()
sns.distplot(bwDF['bandwidth'], ax=ax, kde=False)
plt.show()


fig, ax = plt.subplots()
sns.lineplot(x='B', y='bandwidth', hue='scale', data=bwDF, ax=ax)
fig, ax = plt.subplots()
sns.lineplot(x='B', y='scaledbw', data=bwDF, ax=ax)
