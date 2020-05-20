import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt

# data is aligned to stim onset
# cropEdgesTimes controls the size of the window that is loaded
cropEdgesTimes = [-100e-3, 400e-3]
inputPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202005011400-Peep/emgLoRes/stim/_emg_extraShort_export.h5'
with pd.HDFStore(inputPath, 'r') as store:
    # each trial has its own eesKey, get list of all
    allEESKeys = [
        i
        for i in store.keys()
        if ('stim' in i)]
    # allocate lists to hold data from each trial
    eesList = []
    emgList = []
    for idx, eesKey in enumerate(sorted(allEESKeys)):
        # data for this trial is stored in a pd.dataframe
        stimData = pd.read_hdf(store, eesKey)
        # metadata is stored in a dictionary
        eesMetadata = store.get_storer(eesKey).attrs.metadata
        # extract column names from first trial
        if idx == 0:
            eesColumns = [cn[0] for cn in stimData.columns if cn[1] == 'amplitude']
            emgColumns = [cn[0] for cn in stimData.columns if cn[1] == 'EMG']
            metadataColumns = sorted([k for k in eesMetadata.keys()])
            eesColIdx = [cn for cn in stimData.columns if cn[1] == 'amplitude']
            emgColIdx = [cn for cn in stimData.columns if cn[1] == 'EMG']
            metaDataDF = pd.DataFrame(
                None, index=range(len(allEESKeys)),
                columns=metadataColumns)
        metaDataDF.loc[idx, :] = eesMetadata
        # get mask for requested time points
        cropEdgesMask = (
            (stimData.index >= cropEdgesTimes[0]) &
            (stimData.index <= cropEdgesTimes[1]))
        eesList.append(stimData.loc[cropEdgesMask, eesColIdx])
        emgList.append(stimData.loc[cropEdgesMask, emgColIdx])
# ees information saved in the ees file
eesNP = np.stack(eesList)
# eesNP.shape = trials x time x channel
emgNP = np.stack(emgList)
# eesNP.shape = trials x time x channel
metadataNP = metaDataDF.to_numpy()
# metadataNP.shape = trials x metadata type
# metadata column names are in metadataColumns
# globalIdx is the index of the trial
# combinationIdx is the index of the particular combination
# of rate, active electrodes and amplitude
with pd.HDFStore(inputPath, 'r') as store:
    noiseCeilDF = pd.read_hdf(store, 'noiseCeil').unstack(level='feature')
    columnLabels = noiseCeilDF.columns.to_list()
    electrodeLabels = noiseCeilDF.index.get_level_values('electrode').to_list()
    amplitudeLabels = noiseCeilDF.index.get_level_values('nominalCurrent').to_list()
    covariances = pd.read_hdf(store, 'covariance').unstack(level='feature').to_numpy()
    noiseCeil = noiseCeilDF.to_numpy()
print('finished loading.')
# pdb.set_trace()
checkPlots = True
if checkPlots:
    plt.plot(eesNP[0, :, 0])
    plt.plot(emgNP[0, :, 0])
    plt.show()
    print(metaDataDF.loc[~metaDataDF['outlierTrial'].astype(np.bool), :].groupby(['electrode', 'amplitude'])['RateInHz'].value_counts())
    print(emgList[0].index)
