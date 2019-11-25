import pandas as pd
import numpy as np
import re


def cmpToDF(arrayFilePath):
    arrayMap = pd.read_csv(
        arrayFilePath, sep='\t',
        skiprows=13)
    cmpDF = pd.DataFrame(
        np.nan, index=range(146),
        columns=[
            'xcoords', 'ycoords', 'elecName',
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
        cmpDF.loc[nevIdx, 'label'] = row['label']
        cmpDF.loc[nevIdx, 'bank'] = row['bank']
        cmpDF.loc[nevIdx, 'bankID'] = int(row['elec'])
    cmpDF.dropna(inplace=True)
    return cmpDF


def cmpDFToPrb(
        cmpDF, filePath=None,
        names=None, banks=None,
        contactSpacing=400,  # units of um
        groupIn=None):

    if names is not None:
        keepMask = cmpDF['elecName'].isin(names)
        cmpDF = cmpDF.loc[keepMask, :]
    if banks is not None:
        keepMask = cmpDF['bank'].isin(banks)
        cmpDF = cmpDF.loc[keepMask, :]
    #  
    cmpDF.reset_index(inplace=True, drop=True)
    prbDict = {}
    
    if groupIn is not None:
        groupingCols = []
        for key, spacing in groupIn.items():
            uniqueValues = np.unique(cmpDF[key])
            bins = int(round(
                (uniqueValues.max() - uniqueValues.min() + 1) / spacing))
            cmpDF[key + '_group'] = np.nan
            cmpDF.loc[:, key + '_group'] = pd.cut(
                cmpDF[key], bins, include_lowest=True, labels=False)
            groupingCols.append(key + '_group')
    else:
        groupingCols = ['elecName']
    #  
    for idx, (name, group) in enumerate(cmpDF.groupby(groupingCols)):
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
    """
    tallyChans = []
    tallyGeoms = {}
    for k,v in prbDict.items():
        tallyChans += v['channels']
        tallyGeoms.update(v['geometry'])
    import pdb; 
    """
    if filePath is not None:
        with open(filePath, 'w') as f:
            f.write('channel_groups = ' + str(prbDict))
    return prbDict


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