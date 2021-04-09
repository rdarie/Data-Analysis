import numpy as np
import pywt
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import seaborn as sns

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt


def plotKernels(wav, scales, dt=1):
    # wav = pywt.ContinuousWavelet('cmor1.0-2.0')
    # scales = np.asarray([5, 10, 20, 30])
    # dt = 1e-3
    fs = dt ** (-1)
    frequencies = pywt.scale2frequency(wav, scales) / dt
    # print the range over which the wavelet will be evaluated
    print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
        wav.lower_bound, wav.upper_bound))
    # by default upper_bound and lower_bound are plus minus 8
    # these correspond to the min and max of x below
    width = wav.upper_bound - wav.lower_bound
    max_len = int(np.max(scales)*width + 1)
    max_len_sec = max_len * dt
    # t = np.arange(max_len)
    fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))
    bws = np.zeros(len(scales))
    fcs = np.zeros(len(scales))
    bw_ratios = np.zeros(len(scales))
    # The following code is adapted from the internals of cwt
    # int_psi is scale invariant!
    int_psi, x = pywt.integrate_wavelet(wav, precision=10)
    step = x[1] - x[0]
    for n, scale in enumerate(scales):
        j = np.floor(
            np.arange(scale * width + 1) / (scale * step))
        if np.max(j) >= np.size(int_psi):
            j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        j = j.astype(np.int)
        print(np.diff(j))
        # normalize int_psi for easier plotting
        int_psi /= np.abs(int_psi).max()
        #
        # discrete samples of the integrated wavelet
        # filt = int_psi[j][::-1]
        filt = int_psi[j]
        # filt = int_psi
        #
        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.
        nt = len(filt)
        t = np.linspace(-nt//2, nt//2, nt) * dt
        t_extent = nt * dt
        # print(t.shape)
        axes[n, 0].plot(t, filt.real, t, filt.imag)
        # axes[n, 0].set_xlim([-max_len//2, max_len//2])
        axes[n, 0].set_xlim([-max_len_sec/2, max_len_sec/2])
        axes[n, 0].set_ylim([-1, 1])
        if n != (len(scales) - 1):
            axes[n, 0].set_xticks([])
        axes[n, 0].text(
            max_len_sec/8, 0.1,
            'scale = {:.1f}\nextent = {:.3f} sec'.format(
                scale, t_extent))
        # f = np.linspace(-np.pi, np.pi, max_len)
        f = np.linspace(-fs/2, fs/2, max_len)
        df = f[1] - f[0]
        filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
        filt_fft /= np.abs(filt_fft).max()
        filt_fft_power = np.abs(filt_fft)**2
        H, A, x0, sigma = gauss_fit(f, filt_fft_power)
        fcs[n] = np.abs(x0)
        deltaF = np.abs(x0) - frequencies[n]
        # bws[n] = 2.35482 * sigma
        bws[n] = sigma
        lb = frequencies[n] - bws[n]
        rb = frequencies[n] + bws[n]
        # crosses_half = filt_fft_power > 0.5
        # bws[n] = f[crosses_half][-1] - f[crosses_half][0] + df
        axes[n, 1].plot(f, filt_fft_power)
        #
        axes[n, 1].axvline(lb, c='r')
        axes[n, 1].axvline(rb, c='r')
        axes[n, 1].axvline(frequencies[n], c='g')
        axes[n, 1].axvline(x0, c='m', ls='--')
        #
        axes[n, 1].set_xlim([-fs/2, fs/2])
        axes[n, 1].set_ylim([0, 1])
        if n == (len(scales) - 1):
            axes[n, 1].set_xticks([-fs/2, 0, fs/2])
        else:
            axes[n, 1].set_xticks([])
        # axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        axes[n, 1].grid(True, axis='x')
        axes[n, 1].text(
            frequencies[n] + 2 * bws[n], 0.1,
            'scale = {}\nf_center = {:.1f} Hz (empirical {:.1f} Hz)\nfwhm={:.1f} Hz\ndf={:.1f} Hz\ndeltaF={:.1f}'.format(
                scale, frequencies[n], x0, bws[n], df, deltaF))

    axes[n, 0].set_xlabel('time (sec)')
    axes[n, 1].set_xlabel('frequency (Hz)')
    axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
    axes[0, 1].legend(['Power'], loc='upper left')
    axes[0, 0].set_title('filter')
    axes[0, 1].set_title(r'|FFT(filter)|$^2$')
    fig.suptitle('wavelet {} sampling freq of {} Hz'.format(wav.name, fs))

    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6))
    ax2[0].plot(scales, bws, 'b-', label='bandwidths (empirical)')
    ax2[1].plot(scales, fcs, 'b-', label='center frequencies (empirical)')
    ax2[1].plot(scales, frequencies, 'g-', label='center frequencies (pywt fun)')
    if hasattr(wav, 'bandwidth_frequency'):
        B = wav.bandwidth_frequency
        calcBWs = fs / (2 * np.pi * np.sqrt(B) * scales)
        bw_ratios = bws / calcBWs
        ax2[0].plot(scales, calcBWs, 'r--', label='bandwidths (formula)')
        ax2[0].set_title('{}'.format(bw_ratios))
    if hasattr(wav, 'center_frequency'):
        C = wav.center_frequency
        ax2[1].plot(scales, C * fs / scales, 'r--', label='center frequencies (formula)')
    ax2[1].set_xlabel('scales')
    ax2[0].set_ylabel('Hz')
    ax2[1].set_ylabel('Hz')
    ax2[0].legend()
    ax2[1].legend()
    fig2.suptitle('wavelet {} sampling freq of {} Hz'.format(wav.name, fs))
    plt.show()
    return bws, fcs, bw_ratios


def bandwidthToMorletB(bandwidth, fs=1, scale=1):
    B = (fs / (2 * np.pi * scale * bandwidth)) ** 2
    return B

def centerToMorletC(center, fs=1, scale=1):
    C = (center * scale) / fs
    return C

def freqsToMorletParams(bandwidth, center, fs=1, scale=1):
    B = bandwidthToMorletB(bandwidth, fs=fs, scale=scale)
    C = centerToMorletC(center, fs=fs, scale=scale)
    return B, C

hBound = 8.0
lBound = 1.5
bandwidth = (hBound - lBound) / 2  # Hz
center = (hBound + lBound) / 2  # Hz
# dt = 1e-3
dt = 2e-4
B, C = freqsToMorletParams(bandwidth, center, fs=dt ** -1, scale=500)

bws, fcs, bw_ratios = plotKernels(
    pywt.ContinuousWavelet(
        'cmor{:.3f}-{:.3f}'.format(
            B, C)),
    np.asarray([1, 20, 100, 500]), dt=dt)
# thisWav = pywt.ContinuousWavelet('cmor1.0-1.0')
# theseScales = np.asarray([5, 10, 20, 30])
theseScales = np.linspace(5, 20, 10)
# theseBws, theseFCs, theseBwrs = plotKernels(thisWav, theseScales, dt=1e-3)

bwDict = {}
for thisB in np.arange(2, 10):
    temp = {}
    for thisC in np.arange(1, 3):
        bws, fcs, bw_ratios = plotKernels(
            pywt.ContinuousWavelet('cmor{:.1f}-{:.1f}'.format(thisB, thisC)),
            theseScales, dt=dt)
        temp[thisC] = pd.DataFrame({'bandwidth': bws, 'bw_ratio': bw_ratios}, index=theseScales)
        temp[thisC].index.name = 'scale'
        # plt.close()
    bwDict[thisB] = pd.concat(temp, names='C')
#
bwDF = pd.concat(bwDict, names='B').reset_index()
bwDF.loc[:, 'B'] = bwDF['B'].astype(np.float64) # units of samples ** 2?
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
sns.lineplot(x='B', y='bandwidth', hue='scale', data=bwDF, ax=ax)
fig, ax = plt.subplots()
sns.lineplot(x='B', y='scaledbw', data=bwDF, ax=ax)
