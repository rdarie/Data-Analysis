spectrum[whichChans, :, whichFreqs].shape
spectrum = ns5Data['channel']['spectrum']['PSD']
flatSpectrum = spectrum.transpose(2, 1, 0).to_frame()
flatSpectrum.shape
ylogreg.shape

ns5Data['channel']['spectrum']['t'].shape
dummyVar = np.ones(ns5Data['channel']['spectrum']['t'].shape[0]) * 1
predictedLabels.shape
upMaskSpectrumPredicted.shape
dummyVar[upMaskSpectrumPredicted].shape
dummyVar[downMaskSpectrumPredicted].shape
