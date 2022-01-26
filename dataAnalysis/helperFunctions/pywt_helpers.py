import numpy as np
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import pdb
from scipy.optimize import curve_fit
import pandas as pd
from scipy.interpolate import interp1d

def gauss(x, H, A, x0, sigma):
    return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

def gauss_fit(x, y):
    mean = sum(x * y) / sum(y)
    sigma = np.sqrt(sum(y * (x - mean) ** 2) / sum(y))
    popt, pcov = curve_fit(gauss, x, y, p0=[min(y), max(y), mean, sigma])
    return popt

def getKernel(
        wav, scale, dt=1, precision=10, verbose=False, width=16):
    # test params
    '''
        wav = pywt.ContinuousWavelet('cmor2.0-1.0')
        scales = np.asarray([5, 10, 20, 30, 255])
        dt = 1e-3
        precision = 12
        verbose=True
        width = 8
        '''
    wav.lower_bound = (-1) * width / 2
    wav.upper_bound = width / 2
    # print the range over which the wavelet will be evaluated
    print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
        wav.lower_bound, wav.upper_bound))
    # by default upper_bound and lower_bound are plus minus 8
    # these correspond to the min and max of x below
    # The following code is adapted from the internals of cwt
    # int_psi is scale invariant!
    int_psi, x = pywt.integrate_wavelet(wav, precision=precision)
    #
    step = x[1] - x[0]
    ntpsi = len(int_psi)
    t_psi = np.linspace(-ntpsi // 2, ntpsi // 2, ntpsi) * dt
    #
    # the scale parameter corresponds to downsampling the "mother wavelet", int_psi
    # the size of the convolution kernel is len(j) * dt = (scale * width + 1) * dt
    j = np.floor(
        np.arange(scale * width + 1) / (scale * step))
    assert (scale * step) < 1  # otherwise we fail to downsample int_psi
    if np.max(j) >= np.size(int_psi):
        j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
    j = j.astype(np.int)
    if verbose:
        print('downsampling mother wavelet by:')
        print('{} samples'.format(np.unique(np.diff(j))))
    #
    # discrete samples of the integrated wavelet
    filt = int_psi[j]
    # The CWT consists of convolution of filt with the signal at this scale
    # Here we plot this discrete convolution kernel at each scale.
    nt = len(filt)
    t = np.linspace(-nt // 2, nt // 2, nt) * dt
    t_extent = nt * dt
    pdb.set_trace()
    return t_extent

def plotKernels(
        wav, scales, dt=1, precision=10, verbose=False,
        width=16):
    # test params
    '''
        wav = pywt.ContinuousWavelet('cmor2.0-1.0')
        scales = np.asarray([5, 10, 20, 30, 255])
        dt = 1e-3
        precision = 12
        verbose=True
        width = 8
        '''
    fs = dt ** (-1)
    wav.lower_bound = (-1) * width / 2
    wav.upper_bound = width / 2
    frequencies = pywt.scale2frequency(wav, scales) / dt
    # print the range over which the wavelet will be evaluated
    print("Continuous wavelet will be evaluated over the range [{}, {}]".format(
        wav.lower_bound, wav.upper_bound))
    # by default upper_bound and lower_bound are plus minus 8
    # these correspond to the min and max of x below
    max_len = int(np.max(scales)*width + 1)
    max_len_sec = max_len * dt
    # t = np.arange(max_len)
    bws = np.zeros(len(scales))
    fcs = np.zeros(len(scales))
    bw_ratios = np.zeros(len(scales))
    # The following code is adapted from the internals of cwt
    # int_psi is scale invariant!
    int_psi, x = pywt.integrate_wavelet(wav, precision=precision)
    int_psi_normalized = int_psi / np.abs(int_psi).max()
    #
    # filt_fun = interp1d(x, int_psi)
    step = x[1] - x[0]
    ntpsi = len(int_psi)
    t_psi = np.linspace(-ntpsi//2, ntpsi//2, ntpsi) * dt
    fig_psi, ax_psi = plt.subplots(figsize=(12, 6))
    ax_psi.plot(t_psi, int_psi.real, label='real')
    ax_psi.plot(t_psi, int_psi.imag, label='imag')
    ax_psi.plot(t_psi, np.abs(int_psi), label='absolute value')
    ax_psi.set_title('Raw wavelet')
    ax_psi.set_xlabel('time (sec)')
    ax_psi.legend()
    figsDict = {'psi': (fig_psi, ax_psi)}
    fig, axes = plt.subplots(len(scales), 2, figsize=(12, 6))
    for n, scale in enumerate(scales):
        ## ## assert (scale * step) < 1 # otherwise we fail to downsample int_psi
        ## ## # altj = np.linspace(x[0], x[-1], int(scale * width + 1))
        ## ## # filt = filt_fun(altj)
        ## ## #
        ## ## # the scale parameter corresponds to downsampling the "mother wavelet", int_psi
        ## ## # the size of the convolution kernel is len(j) * dt = (scale * width + 1) * dt
        ## ## j = np.floor(
        ## ##     np.arange(scale * width + 1) / (scale * step))
        ## ## # np.diff(np.arange(scale * width + 1) / (scale * step))
        ## ## if np.max(j) >= np.size(int_psi):
        ## ##     j = np.delete(j, np.where((j >= np.size(int_psi)))[0])
        ## ## j = j.astype(np.int)
        ## ## if verbose:
        ## ##     print('downsampling mother wavelet by:')
        ## ##     print('{} samples'.format(np.unique(np.diff(j))))
        ## ## #
        ## ## # discrete samples of the integrated wavelet
        ## ## # filt = int_psi[j][::-1]
        ## ## filt = int_psi[j]
        ## ## filt_normalized = int_psi_normalized[j]
        ####### new way with interpolation
        # Determine scale-dependent sampling instants
        step = 1.0 / scale
        xs = np.arange(x[0], x[-1] + 0.01 * step, step)
        if xs[-1] > x[-1]:
            xs = xs[:-1]
        # Approximate values by linear interpolation
        filt = np.interp(xs, x, int_psi)
        filt_normalized = np.interp(xs, x, int_psi_normalized)
        ####### new way with interpolation
        # filt = int_psi
        #
        # The CWT consists of convolution of filt with the signal at this scale
        # Here we plot this discrete convolution kernel at each scale.
        nt = len(filt)
        t = np.linspace(-nt//2, nt//2, nt) * dt
        t_extent = nt * dt
        # print(t.shape)
        _, _, x0_t, standard_dev_t = gauss_fit(t, np.abs(filt))
        rb_t = standard_dev_t
        lb_t = - standard_dev_t
        axes[n, 0].plot(t, filt_normalized.real, t, filt_normalized.imag)
        axes[n, 0].axvline(lb_t, c='r')
        axes[n, 0].axvline(rb_t, c='r')
        if hasattr(wav, 'bandwidth_frequency'):
            B = wav.bandwidth_frequency
            rb_t_theoretical = scale * dt * np.sqrt(B / 2)
            axes[n, 0].axvline(rb_t_theoretical, c='c', ls='--')
            axes[n, 0].axvline(-1 * rb_t_theoretical, c='c', ls='--')
        # axes[n, 0].set_xlim([-max_len//2, max_len//2])
        axes[n, 0].set_xlim([-max_len_sec/2, max_len_sec/2])
        axes[n, 0].set_ylim([-1, 1])
        if n != (len(scales) - 1):
            axes[n, 0].set_xticks([])
        axes[n, 0].text(
            max_len_sec/8, 0.1,
            'scale = {:.1f}\nextent = {:.3f} sec'.format(
                scale, rb_t_theoretical * 2))
        # f = np.linspace(-np.pi, np.pi, max_len)
        f = np.linspace(-fs/2, fs/2, max_len)
        df = f[1] - f[0]
        filt_fft = np.fft.fftshift(np.fft.fft(filt, n=max_len))
        filt_fft_abs = np.abs(filt_fft)
        H, A, x0, standard_dev = gauss_fit(f, filt_fft_abs)
        bws[n] = np.abs(standard_dev)
        fcs[n] = np.abs(x0)
        deltaF = np.abs(x0) - frequencies[n]
        lb = x0 - bws[n]
        rb = x0 + bws[n]
        # normalize
        filt_fft_normalized = filt_fft_abs / filt_fft_abs.max()
        filt_fft_power = np.abs(filt_fft)**2
        # lb = frequencies[n] - bws[n]
        # rb = frequencies[n] + bws[n]
        # crosses_half = filt_fft_power > 0.5
        # bws[n] = f[crosses_half][-1] - f[crosses_half][0] + df
        axes[n, 1].plot(f, filt_fft_normalized)
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
            axes[n, 1].set_xticks([0])
        # axes[n, 1].set_xticklabels([r'$-\pi$', '0', r'$\pi$'])
        axes[n, 1].grid(True, axis='x')
        axes[n, 1].text(
            frequencies[n] + 2 * bws[n], 0.1,
            'scale = {}\nf_c = {:.1f} Hz\n  (empirical f_c {:.1f} Hz)\nfwhm={:.1f} Hz\ndf={:.1f} Hz\ndeltaF={:.1f}'.format(
                scale, frequencies[n], x0, bws[n], df, deltaF))
    #
    axes[n, 0].set_xlabel('time (sec)')
    axes[n, 1].set_xlabel('frequency (Hz)')
    axes[0, 0].legend(['real', 'imaginary'], loc='upper left')
    axes[0, 1].legend(['Normalized power'], loc='upper left')
    axes[0, 0].set_title('filter')
    axes[0, 1].set_title(r'|FFT(filter)|$^2$')
    fig.suptitle('wavelet {} sampling freq of {} Hz'.format(wav.name, fs))
    figsDict['scales'] = (fig, axes)
    fig2, ax2 = plt.subplots(2, 1, figsize=(12, 6))
    ax2[0].plot(scales, bws, 'b-', label='bandwidths (empirical)')
    ax2[1].plot(scales, fcs, 'b-', label='center frequencies (empirical)')
    ax2[1].plot(scales, frequencies, 'g-', label='center frequencies (pywt fun)')
    if hasattr(wav, 'bandwidth_frequency'):
        B = wav.bandwidth_frequency
        calcBWs = fs / (np.pi * np.sqrt(2 * B) * scales)
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
    figsDict['bandwidth'] = (fig2, ax2)
    return bws, fcs, bw_ratios, figsDict

def morletBtoBandwidth(B, fs=1, scale=1):
    bandwidth = fs * (np.pi * scale * np.sqrt(2 * B)) ** (-1)
    return bandwidth

def bandwidthToMorletB(bandwidth, fs=1, scale=1):
    B = (fs / (np.pi * scale * bandwidth)) ** 2 / 2
    return B

def centerToMorletC(center, fs=1, scale=1):
    C = (center * scale) / fs
    return C

def bandwidthToMorletSigma(bandwidth, fs=1, scale=1):
    sigma = fs / (2 * np.pi * scale * bandwidth)
    return sigma

def freqsToMorletParams(bandwidth, center, fs=1, scale=1):
    B = bandwidthToMorletB(bandwidth, fs=fs, scale=scale)
    C = centerToMorletC(center, fs=fs, scale=scale)
    return B, C

def morletBToSigma(B, fs=1, scale=1):
    sig = scale * np.sqrt(B / 2) / fs
    return sig