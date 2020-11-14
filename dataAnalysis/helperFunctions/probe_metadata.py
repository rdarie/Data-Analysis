import pandas as pd
import numpy as np
import re
from datetime import datetime as dt
import os
import pdb

def getLatestImpedance(
        recordingDate=None, impedanceFilePath='./impedances.h5',
        recordingDateStr=None,
        block=None, elecType=None):
    impedances = pd.read_hdf(impedanceFilePath, 'impedance')
    if elecType is not None:
        impedances = impedances.query('elecType == "{}"'.format(elecType))
    if recordingDate is None:
        if recordingDateStr is not None:
            recordingDate = dt.strptime(recordingDateStr, '%Y%m%d%H%M')
        else:
            recordingDate = block.rec_datetime
    pastDates = impedances.loc[impedances['date'] <= recordingDate, 'date']
    lastDate = np.max(pastDates)
    impedances = impedances.loc[impedances['date'] == lastDate, :]
    return impedances


def cmpToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='\t',
        skiprows=13)
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'zcoords', 'elecName',
            'elecID', 'label', 'bank', 'bankID', 'nevID']
        )
    bankLookup = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5}
    for rowIdx, row in arrayMap.iterrows():
        #  label matches non-digits index matches digit
        elecSplit = re.match(r'(\D*)(\d*)', row['label']).groups()
        elecName = elecSplit[0]
        elecIdx = int(elecSplit[-1])
        nevIdx = int(row['elec']) + bankLookup[row['bank']] * 32
        cmpDF.loc[nevIdx, 'elecID'] = elecIdx
        cmpDF.loc[nevIdx, 'nevID'] = nevIdx
        cmpDF.loc[nevIdx, 'elecName'] = elecName
        cmpDF.loc[nevIdx, 'xcoords'] = row['row']
        cmpDF.loc[nevIdx, 'ycoords'] = row['//col']
        cmpDF.loc[nevIdx, 'zcoords'] = 0
        cmpDF.loc[nevIdx, 'label'] = row['label']
        cmpDF.loc[nevIdx, 'bank'] = row['bank']
        cmpDF.loc[nevIdx, 'bankID'] = int(row['elec'])
    cmpDF.dropna(inplace=True)
    cmpDF.reset_index(drop=True, inplace=True)
    return cmpDF


def mapToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='; ',
        skiprows=10, header=None, engine='python',
        names=['FE', 'electrode', 'position'])
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'zcoords', 'elecName',
            'elecID', 'label', 'bank', 'bankID', 'nevID']
        )
    bankLookup = {'A.1': 0, 'A.2': 1, 'A.3': 2}
    for rowIdx, row in arrayMap.iterrows():
        processor, port, FEslot, channel = row['FE'].split('.')
        bankName = '{}.{}'.format(port, FEslot)
        array, electrodeFull = row['electrode'].split('.')
        if '_' in electrodeFull:
            electrode, electrodeRep = electrodeFull.split('_')
        else:
            electrode = electrodeFull
        x, y, z = row['position'].split('.')
        nevIdx = int(channel) - 1 + bankLookup[bankName] * 32
        cmpDF.loc[nevIdx, 'elecID'] = int(electrode[1:])
        cmpDF.loc[nevIdx, 'nevID'] = nevIdx
        cmpDF.loc[nevIdx, 'elecName'] = array
        cmpDF.loc[nevIdx, 'xcoords'] = float(x)
        cmpDF.loc[nevIdx, 'ycoords'] = float(y)
        cmpDF.loc[nevIdx, 'zcoords'] = float(z)
        cmpDF.loc[nevIdx, 'label'] = row['electrode'].replace('.', '_')
        cmpDF.loc[nevIdx, 'bank'] = bankName
        cmpDF.loc[nevIdx, 'bankID'] = int(channel)
        cmpDF.loc[nevIdx, 'FE'] = row['FE']
    #
    cmpDF.dropna(inplace=True)
    cmpDF.reset_index(inplace=True, drop=True)
    # import pdb; pdb.set_trace()
    # xIdx = np.array(
    #     cmpDF['xcoords'].values - cmpDF['xcoords'].min(),
    #     dtype=np.int)
    # yIdx = np.array(
    #     cmpDF['ycoords'].values - cmpDF['ycoords'].min(),
    #     dtype=np.int)
    cmpDF.loc[:, 'nevID'] += 1
    return cmpDF


def cmpDFToPrb(
        cmpDF, filePath=None,
        names=None, banks=None, labels=None,
        contactSpacing=400,  # units of um
        groupIn=None, verbose=True):
    # pdb.set_trace()
    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]
    if labels is not None:
        keepMask = cmpDF['label'].isin(labels)
        cmpDF = cmpDF.loc[keepMask, :]
    #  
    prbDict = {}
    if groupIn is not None:
        groupingCols = []
        for key, bins in groupIn.items():
            # uniqueValues = np.unique(cmpDF[key])
            # bins = int(round(
            #     (uniqueValues.max() - uniqueValues.min() + 1) / spacing))
            cmpDF.loc[:, key + '_group'] = np.nan
            cmpDF.loc[:, key + '_group'] = pd.cut(
                cmpDF[key], bins, labels=False)
            groupingCols.append(key + '_group')
    else:
        groupingCols = ['elecName']
    if verbose:
        print('Writing prb file (in cmpDFToPrb)....')
    for idx, (name, group) in enumerate(cmpDF.groupby(groupingCols)):
        if verbose:
            print('channel group idx: {} name: {}'.format(idx, name))
        theseChannels = []
        theseGeoms = {}
        for rName, row in group.iterrows():
            theseChannels.append(int(rName))
            theseGeoms.update({
                int(rName): (
                    contactSpacing * row['xcoords'],
                    contactSpacing * row['ycoords'])})
        prbDict.update({idx: {
            'channels': theseChannels,
            'geometry': theseGeoms
            }})
    # pdb.set_trace()
    """
    tallyChans = []
    tallyGeoms = {}
    for k,v in prbDict.items():
        tallyChans += v['channels']
        tallyGeoms.update(v['geometry'])
    import pdb; 
    """
    if filePath is not None:
        if os.path.exists(filePath):
            os.remove(filePath)
        with open(filePath, 'w') as f:
            f.write('channel_groups = ' + str(prbDict))
    return prbDict

'''
def cmpDFToPrbAddDummies(
        cmpDF, filePath=None,
        names=None, banks=None,
        contactSpacing=400,  # units of um
        groupIn=None,
        prependDummy=0, appendDummy=0):

    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]
    #  import pdb; 
    cmpDF.reset_index(inplace=True, drop=True)
    prbDict = {}

    if prependDummy > 0:
        prbDict.update({0: {
            'channels': list(range(prependDummy)),
            'geometry': {
                dummyIdx: (
                    contactSpacing * dummyIdx,
                    contactSpacing * dummyIdx) for dummyIdx in range(prependDummy)}
        }})
        idxOffset = 1
    else:
        idxOffset = 0

    if groupIn is not None:
        groupingCols = []
        for key, spacing in groupIn.items():
            uniqueValues = np.unique(cmpDF[key])
            bins = int(round(len(uniqueValues) / spacing))
            cmpDF[key + '_group'] = np.nan
            cmpDF.loc[:, key + '_group'] = pd.cut(
                cmpDF[key], bins, include_lowest=True, labels=False)
            groupingCols.append(key + '_group')
    else:
        groupingCols = ['elecName']
    for idx, (name, group) in enumerate(cmpDF.groupby(groupingCols)):
        #  import pdb; 
        #  group['nevID'].astype(int).values
        prbDict.update({idx + idxOffset: {
            'channels': list(
                group.index +
                int(prependDummy)),
            'geometry': {
                int(rName) + int(prependDummy): (
                    contactSpacing * row['xcoords'],
                    contactSpacing * row['ycoords']) for rName, row in group.iterrows()}
        }})
        lastChan = group.index[-1] + int(prependDummy)

    if appendDummy > 0:
        #  
        appendChanList = list(range(lastChan + 1, lastChan + appendDummy + 1))
        prbDict.update({idx + idxOffset + 1: {
            'channels': appendChanList,
            'geometry': {
                dummyIdx: (
                    contactSpacing * dummyIdx,
                    contactSpacing * dummyIdx) for dummyIdx in appendChanList}
        }})

    if filePath is not None:
        with open(filePath, 'w') as f:
            f.write('channel_groups = ' + str(prbDict))
    return prbDict
'''