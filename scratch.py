import numpy as np
type(spikes)
spikes.keys()

spikes['NEUEVWAV_HeaderIndices']
type(spikes['TimeStamps'])
len(spikes['TimeStamps'])
len(spikes['TimeStamps'][1])
print(spikes['Units'])
type(spikes['ChannelID'])
type(spikes['Waveforms'][0])
spikes['Waveforms'][1].shape
len(spikes['Classification'][1])
len(spikes['NEUEVWAV_HeaderIndices'])
spike_headers
nev_file = NevFile(datafile)
dir(nev_file)
ch_idx      = spikes['ChannelID'].index(1)
units       = sorted(list(set(spikes['Classification'][ch_idx])))
j = 0

badMask[4].any()
spikes['Waveforms'][4].shape
len(spikes['TimeStamps'][4])
len(spikes['Classification'][4])

bla = 4
np.array(spikes['TimeStamps'][4])[np.logical_not(badMask[bla])].shape
