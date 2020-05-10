import numpy as np
import scipy.io as sio


def saveRasterForNeuralTraj(alignedRastersDF, filePath):
    alignedRasterList = [
        g.to_numpy(dtype='uint8')
        for n, g in alignedRastersDF.groupby(['segment', 'originalIndex'])]
    trialIDs = [
        np.atleast_2d(i).astype('uint16')
        for i in range(len(alignedRasterList))]
    structDType = np.dtype([('trialId', 'O'), ('spikes', 'O')])

    dat = np.array(list(zip(trialIDs, alignedRasterList)), dtype=structDType)
    sio.savemat(filePath, {'dat': dat})
