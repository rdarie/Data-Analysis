import pandas as pd
import numpy as np
import pdb
import matplotlib.pyplot as plt

# data is aligned to stim onset
# cropEdgesTimes controls the size of the window that is loaded
cropEdgesTimes = [-100e-3, 400e-3]
# cropEdgesTimes = [-600e-3, -100e-3]
# inputPath = 'G:\\Delsys\\scratch\\202009231400-Peep\\default\\stim\\_emg_XS_export.h5'
#
inputPath = 'G:\\Delsys\\scratch\\202010191100-Peep\\default\\stim\\_emg_XS_export.h5'
inputPath = 'G:\\Delsys\\scratch\\202007011300-Peep\\_emg_XS_export_0701.h5'
inputPath = 'G:\\Delsys\\scratch\\202010191100-Peep\\parameter_recovery\\stim\\_emg_XS_export.h5'
#
# inputPath = '/gpfs/scratch/rdarie/rdarie/Neural Recordings/202009231400-Peep/default/stim/_emg_XS_export.h5'

with pd.HDFStore(inputPath, 'r') as store:
    # each trial has its own eesKey, get list of all
    allEESKeys = [
        i
        for i in store.keys()
        if ('stim' in i)]
    # allocate lists to hold data from each trial
    eesList = []
    emgList = []
    accList = []
    lfpList = []
    for idx, eesKey in enumerate(sorted(allEESKeys)):
        # print(eesKey)
        # data for this trial is stored in a pd.dataframe
        stimData = pd.read_hdf(store, eesKey)
        # print((stimData.abs() > 0).any())
        # metadata is stored in a dictionary
        eesMetadata = store.get_storer(eesKey).attrs.metadata
        # extract column names from first trial
        if idx == 0:
            eesColumns = [cn[0] for cn in stimData.columns if cn[1] == 'amplitude']
            emgColumns = [cn[0] for cn in stimData.columns if cn[1] == 'emg_env']
            accColumns = [cn[0] for cn in stimData.columns if 'acc' in cn[1]]
            lfpColumns = [cn[0] for cn in stimData.columns if cn[1] == 'lfp']
            metadataColumns = sorted([k for k in eesMetadata.keys()])
            eesColIdx = [cn for cn in stimData.columns if cn[1] == 'amplitude']
            emgColIdx = [cn for cn in stimData.columns if cn[1] == 'emg_env']
            accColIdx = [cn for cn in stimData.columns if 'acc' in cn[1]]
            lfpColIdx = [cn for cn in stimData.columns if cn[1] == 'lfp']
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
        accList.append(stimData.loc[cropEdgesMask, accColIdx])
        lfpList.append(stimData.loc[cropEdgesMask, lfpColIdx])

# ees information saved in the ees variable
eesNP = np.stack(eesList)
# eesNP.shape = trials x time x channel

print('EMG names are:')
print(emgColIdx)
emgNP = np.stack(emgList)
# emgNP.shape = trials x time x channel

print('Acc names are:')
print(accColIdx)
accNP = np.stack(accList)
# accNP.shape = trials x time x channel

print('LFP names are:')
print(lfpColIdx)
lfpNP = np.stack(lfpList)
# lfpNP.shape = trials x time x channel

metadataNP = metaDataDF.to_numpy()
# metadataNP.shape = trials x metadata type
# metadata column names are in metadataColumns
# globalIdx is the index of the trial
# combinationIdx is the index of the particular combination
# of rate, active electrodes and amplitude

with pd.HDFStore(inputPath, 'r') as store:
    if 'noiseCeil' in store:
        noiseCeilDF = pd.read_hdf(store, 'noiseCeil').unstack(level='feature')
        noiseCeilDF.index.set_names('amplitude', level='nominalCurrent', inplace=True)
        columnLabels = noiseCeilDF.columns.to_list()
        electrodeLabels = noiseCeilDF.index.get_level_values('electrode').to_list()
        amplitudeLabels = noiseCeilDF.index.get_level_values('amplitude').to_list()
        noiseCeil = noiseCeilDF.to_numpy()
        noiseCeilMeta = noiseCeilDF.index.to_frame(index=False)

        def getEESIdx(metaRow):
            findMask = (noiseCeilMeta['electrode'] == metaRow['electrode']) & (noiseCeilMeta['RateInHz'] == metaRow['RateInHz']) & (noiseCeilMeta['amplitude'] == metaRow['amplitude'])
            if not noiseCeilMeta.index[findMask].empty:
                return noiseCeilMeta.index[findMask][0]
            else:
                return np.nan

        metaDataDF['eesIdx'] = metaDataDF.apply(getEESIdx, axis=1)
    else:
        noiseCeilDF = None
        noiseCeil = None
        noiseCeilMeta = None
    if 'covariance' in store:
        covariances = pd.read_hdf(store, 'covariance').unstack(level='feature').to_numpy()
    else:
        covariances = None

if 'outlierTrial' not in metaDataDF:
    metaDataDF.loc[:, 'outlierTrial'] = False

print('Number of groups that are not exclusively outliers: ')
print(metaDataDF.loc[~metaDataDF['outlierTrial'], :].groupby(['electrode', 'amplitude', 'RateInHz']).ngroups)

if noiseCeilDF is not None:
    print('Nans in noiseCeil: ')
    nansInNoiseCeilMask = noiseCeilDF.isna().any(axis=1)
    noiseCeilDF.loc[nansInNoiseCeilMask, :]

trialCountGood = metaDataDF.loc[~metaDataDF['outlierTrial'].astype(np.bool), :].groupby(['electrode', 'amplitude'])['RateInHz'].value_counts()
trialCount = metaDataDF.groupby(['electrode', 'amplitude'])['RateInHz'].value_counts()
checkPlots = True
if checkPlots:
    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].plot(eesNP[0, :, 0], label='ees')
    ax[0].legend()
    ax[1].plot(emgNP[0, :, 0], label='emg')
    ax[1].legend()
    try:
        ax[2].plot(accNP[0, :, 0], label='acc')
        ax[2].legend()
    except:
        pass
    try:
        ax[3].plot(lfpNP[0, :, 0], label='lfp')
        ax[3].legend()
    except:
        pass
    plt.show()
    print('Number of trials per ees condition: ')
    print(trialCount)
    print(emgList[0].index)

print('finished loading.')
pdb.set_trace()