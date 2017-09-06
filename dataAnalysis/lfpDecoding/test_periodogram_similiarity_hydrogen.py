
channelData = clean_data
fr_cutoff = 600

channelData['data'].shape
signal = channelData['data'][0]
signal.shape
len(P_scipy)

import matplotlib.colors as colors
P_libtfr.shape
plt.pcolormesh(P_libtfr[0,:,:], norm = colors.SymLogNorm(linthresh=0.01, linscale=0.01,
    vmin=P_libtfr.min().min(), vmax=P_libtfr.max().max()))
plt.show()

P_scipy.shape
plt.pcolormesh(P_scipy[0,:,:],norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
        vmin=P_scipy.min().min(), vmax=P_scipy.max().max()))
plt.show()

NFFT / Fs * 0.5
t_scipy = P_scipy[1] + channelData['start_time_s']
t_scipy
t+3
t_scipy[-1] + stepLen_s + NFFT / Fs * 0.5
channelData['t'][-1]
D.nfft
Np = 201
K = 6
tm = 6.0
flock = 0.01
tlock = 5
S = libtfr.tfr_spec(signal, NFFT, stepLen_samp, Np, K, tm, flock, tlock)
S.shape
S = S[np.newaxis,:,:fr_samp]
plt.pcolormesh(S[0,:,:],norm=colors.SymLogNorm(linthresh=0.01, linscale=0.01,
        vmin=S.min().min(), vmax=S.max().max()))
plt.show()
Z = D.mtstft(signal, stepLen_samp)
Z.shape
