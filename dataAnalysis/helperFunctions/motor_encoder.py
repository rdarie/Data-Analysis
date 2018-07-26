import pdb
from dataAnalysis.helperFunctions.helper_functions import *
import pandas as pd

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

import seaborn as sns
from scipy.signal import hilbert
import line_profiler

#@profile
def debounceLine(dataLine):
    pass

#@profile
def getTransitionIdx(motorData, edgeType = 'rising'):

    if edgeType == 'rising':
        Adiff = motorData['A_int'].diff()
        Bdiff = motorData['B_int'].diff()

        transitionMask = np.logical_or(Adiff == 1, Bdiff == 1)
    elif edgeType == 'falling':
        Adiff = motorData['A_int'].diff()
        Bdiff = motorData['B_int'].diff()

        transitionMask = np.logical_or(Adiff == -1, Bdiff == -1)
    elif edgeType == 'both':
        Adiff = motorData['A_int'].diff().abs()
        Bdiff = motorData['B_int'].diff().abs()

    transitionMask = np.logical_or(Adiff == 1, Bdiff == 1)

    transitionIdx = motorData.index[transitionMask].tolist()

    return transitionMask, transitionIdx

#@profile
def getMotorData(ns5FilePath, inputIDs, startTime, dataTime, debounce = None):

    analogData = getNSxData(ns5FilePath, inputIDs.values(), startTime, dataTime)
    newColNames = {num : '' for num in analogData['data'].columns}

    for key, value in inputIDs.items():
        idx = analogData['elec_ids'].index(value)
        newColNames[idx] = key

    motorData = analogData['data'].rename(columns = newColNames)
    motorData['A'] = motorData['A+'] - motorData['A-']
    motorData['B'] = motorData['B+'] - motorData['B-']
    motorData['Z'] = motorData['Z+'] - motorData['Z-']

    motorData.drop(['A+', 'A-', 'B+', 'B-', 'Z+', 'Z-'], axis = 1, inplace = True)

    """
    # use gaussian mixture model to digitize analog trace
    levelEstimators = {name : GaussianMixture(n_components=2, covariance_type= 'spherical', max_iter=200, random_state=0) for name in motorData.columns}
    for key, estimator in levelEstimators.items():
        estimator.fit(motorData[key].values.reshape(-1, 1))
        motorData.loc[:,key + 'int'] = levelEstimators[key].predict(motorData[key].values.reshape(-1,1))
        if debounce is not None:
            motorData.loc[:,key] = debounceLine(dataLine, debounce)
    """
    """
    # use signal range to digitize analog trace
    """
    for column in motorData.columns:
        threshold = (motorData[column].max() - motorData[column].min() ) / 2
        motorData.loc[:,column + '_int'] = (motorData[column] > threshold).astype(int)

    transitionMask, transitionIdx = getTransitionIdx(motorData, edgeType = 'both')
    motorData['encoderState'] = 0
    motorData['count'] = 0
    state1Mask = np.logical_and(motorData['A_int'] == 1, motorData['B_int'] == 1)
    motorData.loc[state1Mask, 'encoderState'] = 1
    state2Mask = np.logical_and(motorData['A_int'] == 1, motorData['B_int'] == 0)
    motorData.loc[state2Mask, 'encoderState'] = 2
    state3Mask = np.logical_and(motorData['A_int'] == 0, motorData['B_int'] == 0)
    motorData.loc[state3Mask, 'encoderState'] = 3
    state4Mask = np.logical_and(motorData['A_int'] == 0, motorData['B_int'] == 1)
    motorData.loc[state4Mask, 'encoderState'] = 4

    incrementLookup = {
            (1, 0) : 0,
            (1, 1) : 0,
            (1, 2) : +1,
            (1, 3) : 0,
            (1, 4) : -1,
            (2, 0) : 0,
            (2, 1) : -1,
            (2, 2) : 0,
            (2, 3) : +1,
            (2, 4) : 0,
            (3, 0) : 0,
            (3, 1) : 0,
            (3, 2) : -1,
            (3, 3) : 0,
            (3, 4) : +1,
            (4, 0) : 0,
            (4, 1) : +1,
            (4, 2) : 0,
            (4, 3) : -1,
            (4, 4) : 0,
            }

    statesAtTransition = motorData.loc[transitionIdx, 'encoderState'].tolist()
    transitionStatePairs = [(statesAtTransition[i], statesAtTransition[i-1]) for i in range(1, len(statesAtTransition))]
    count = [incrementLookup[pair] for pair in transitionStatePairs]
    #pad with a zero to make up for the fact that the first one doesn't have a pair
    motorData.loc[transitionIdx,'count'] = [0] + count
    #pdb.set_trace()
    motorData['position'] = motorData['count'].cumsum()
    #motorData['stateChanges'] = motorData['encoderState'].diff().abs() != 0
    #pdb.set_trace()

    return motorData

#@profile
def getTrials(motorData, trialType = '2AFC'):

    rightLEDdiff = motorData['rightLED_int'].diff()
    rightButdiff = motorData['rightBut_int'].diff()
    leftLEDdiff = motorData['leftLED_int'].diff()
    leftButdiff = motorData['leftBut_int'].diff()

    #pdb.set_trace()
    rightLEDOnsetIdx = motorData.index[rightLEDdiff == -1].tolist()
    rightButEdges = rightButdiff == 1
    rightButOnsetIdx = []

    leftLEDOnsetIdx = motorData.index[leftLEDdiff == -1].tolist()
    leftButEdges = leftButdiff == 1
    leftButOnsetIdx = []

    transitionMask, transitionIdx = getTransitionIdx(motorData, edgeType = 'rising')
    trialEvents = pd.DataFrame(index = [], columns = ['Time', 'Label', 'Details'])

    #slow detection of movement onset and offset
    movementOnsetIdx = []
    movementOffsetIdx = []
    # look at this size window and determine if there's an index there
    windowLen = int(100e-3 / 3e-4)
    # How many transitions should be in the window in order for this to count as a "start"
    moveThreshold = 10
    # at most how many transitions can be in the idx for us not to count it?
    noMoveThreshold = 2
    # once we find a start, how many indices is it safe to skip ahead?
    skipAheadInc = 1000

    firstIdx = motorData.index[0]
    lastIdx = motorData.index[-1]
    lookForStarts = True

    transitionIdxIter = iter(transitionIdx)
    for idx in transitionIdxIter:
        #check that we're not out of bounds])
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
            #old way, index into transition mask and find the number of crossings
            transitionInPast = sum(transitionMask[idx - windowLen : idx])
            transitionInFuture = sum(transitionMask[idx : idx + windowLen])

            if transitionInPast < noMoveThreshold and transitionInFuture > moveThreshold and lookForStarts:
                movementOnsetIdx.append(idx)
                trialEvents = trialEvents.append({'Time':idx,'Label':'Movement Onset'}, ignore_index = True)
                for _ in range(skipAheadInc):
                    next(transitionIdxIter, None)
                lookForStarts = False
            if transitionInPast > moveThreshold and transitionInFuture < noMoveThreshold and not lookForStarts:
                movementOffsetIdx.append(idx)
                trialEvents = trialEvents.append({'Time':idx,'Label':'Movement Offset'}, ignore_index = True)
                lookForStarts = True

    if len(movementOnsetIdx) > len(movementOffsetIdx):
        movementOnsetIdx = movementOnsetIdx[:-1]

    assert len(movementOnsetIdx) == len(movementOffsetIdx)

    # detection of right button choice (debouce)
    #skipAhead = 0
    # look at this size window and determine if there's an index there
    windowLen = int(200e-3 / 3e-4)
    # How many transitions should be in the window in order for this to count as a "start"
    onThreshold = 20
    # at most how many transitions can be in the idx for us not to count it?
    offThreshold = 2
    for idx in motorData.index[rightButEdges].tolist():
        #check that we're not out of bounds])
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
    #        if not skipAhead:
            transitionInPast = sum(motorData['rightBut_int'][idx - windowLen : idx])
            transitionInFuture = sum(motorData['rightBut_int'][idx : idx + windowLen])
            if transitionInPast < offThreshold and transitionInFuture > onThreshold:
                rightButOnsetIdx.append(idx)
                trialEvents = trialEvents.append({'Time':idx,'Label':'Right Button Onset'}, ignore_index = True)
    #            skipAhead = skipAheadInc
    #        else:
    #            skipAhead = skipAhead - 1

    #skipAhead = 0
    for idx in motorData.index[leftButEdges].tolist():
        #check that we're not out of bounds])
        #if idx > 161400 and idx < 161600:
        #    pdb.set_trace()
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
    #        if not skipAhead:
            transitionInPast = sum(motorData['leftBut_int'][idx - windowLen : idx])
            transitionInFuture = sum(motorData['leftBut_int'][idx : idx + windowLen])
            if transitionInPast < offThreshold and transitionInFuture > onThreshold:
                leftButOnsetIdx.append(idx)
                trialEvents = trialEvents.append({'Time':idx,'Label':'Left Button Onset'}, ignore_index = True)
    #            skipAhead = skipAheadInc
    #        else:
    #            skipAhead = skipAhead - 1

    for idx in rightLEDOnsetIdx:
        trialEvents = trialEvents.append({'Time':idx,'Label':'Right LED Onset'}, ignore_index = True)
    for idx in leftLEDOnsetIdx:
        trialEvents = trialEvents.append({'Time':idx,'Label':'Left LED Onset'}, ignore_index = True)


    trialEvents.sort_values('Time', inplace = True)
    trialEvents.reset_index(drop=True, inplace = True)
    if trialType == '2AFC':
        while not (trialEvents.loc[:3,'Label'].str.contains('Movement').all()) or trialEvents.loc[4:4,'Label'].str.contains('Movement').all():
            #above expression is not true if the first 5 events do not make up a complete sequence
            #pdb.set_trace()
            trialEvents.drop(0, inplace = True)
            trialEvents.reset_index(drop=True, inplace = True)

        #pdb.set_trace()
        trialStartIdx = trialEvents.index[trialEvents['Label'] == 'Movement Onset']
        trialStartIdx = trialStartIdx[range(0,len(trialStartIdx),2)]
        eventsToTrack = ['First',
                         'FirstOnset',
                         'FirstOffset',
                         'Second',
                         'SecondOnset',
                         'SecondOffset',
                         'Magnitude',
                         'Direction',
                         'Condition',
                         'Type',
                         'Stimulus Duration',
                         'Outcome',
                         'Choice',
                         'CueOnset',
                         'ChoiceOnset'
                         ]
        trialStats = pd.DataFrame(index = range(len(trialStartIdx)), columns = eventsToTrack)

        for idx, startIdx in enumerate(trialStartIdx):
            try:
                try:
                    nextStartIdx = trialStartIdx[idx + 1]
                except:
                    nextStartIdx = trialStartIdx[-1]

                assert trialEvents.loc[startIdx, 'Label'] == 'Movement Onset' and trialEvents.loc[startIdx + 1, 'Label'] == 'Movement Offset'

                offsetMask = trialEvents.loc[startIdx:nextStartIdx,'Label'] == 'Movement Offset'
                offsetTimes = trialEvents.loc[startIdx:nextStartIdx,'Time'][offsetMask]

                firstOnsetTime = trialEvents.loc[startIdx, 'Time']
                trialStats.loc[idx, 'FirstOnset'] = firstOnsetTime
                firstOffsetTime = offsetTimes.iloc[0]
                trialStats.loc[idx, 'FirstOffset'] = firstOffsetTime
                firstMovement = motorData.loc[firstOnsetTime:firstOffsetTime, 'position']

                if abs(firstMovement.max()) < abs(firstMovement.min()):
                    trialStats.loc[idx, 'Direction'] = 'Extension'
                    trialStats.loc[idx, 'First'] = firstMovement.min()
                else:
                    trialStats.loc[idx, 'Direction'] = 'Flexion'
                    trialStats.loc[idx, 'First'] = firstMovement.max()

                assert trialEvents.loc[startIdx + 2, 'Label'] == 'Movement Onset'
                secondOnsetTime = trialEvents.loc[startIdx + 2, 'Time']
                trialStats.loc[idx, 'SecondOnset'] = secondOnsetTime
                secondOffsetTime = offsetTimes.iloc[1]
                trialStats.loc[idx, 'SecondOffset'] = secondOffsetTime
                secondMovement = motorData.loc[secondOnsetTime:secondOffsetTime, 'position']

                if trialStats.loc[idx, 'Direction'] == 'Extension':
                    assert abs(secondMovement.max()) < abs(secondMovement.min())
                    trialStats.loc[idx, 'Second'] = secondMovement.min()
                else:
                    assert abs(secondMovement.max()) > abs(secondMovement.min())
                    trialStats.loc[idx, 'Second'] = secondMovement.max()

                trialStats.loc[idx, 'Magnitude'] = 'Short' if abs(trialStats.loc[idx, 'Second'] - trialStats.loc[idx, 'First']) < 2e4 else 'Long'
                cueMask = trialEvents.loc[startIdx:nextStartIdx, 'Label'].str.contains('LED')
                nCues = sum(cueMask)
                trialStats.loc[idx, 'Condition'] = 'hard' if nCues == 2 else 'easy'
                trialStats.loc[idx, 'CueOnset'] = trialEvents.loc[startIdx:nextStartIdx, 'Time'][cueMask].mean()

                trialStats.loc[idx, 'Type'] = 0 if abs(trialStats.loc[idx, 'First']) < abs(trialStats.loc[idx, 'Second']) else 1
                trialStats.loc[idx, 'Stimulus Duration'] = secondOffsetTime - firstOnsetTime

                rightButtonMask = trialEvents.loc[startIdx:nextStartIdx, 'Label'].str.contains('Right Button')
                leftButtonMask = trialEvents.loc[startIdx:nextStartIdx, 'Label'].str.contains('Left Button')
                if rightButtonMask.any() and leftButtonMask.any():
                    trialStats.loc[idx, 'Choice'] = 'both'
                    trialStats.loc[idx, 'Outcome'] = 'incorrect button'
                elif rightButtonMask.any():
                    trialStats.loc[idx, 'Choice'] = 'right'
                    trialStats.loc[idx, 'ChoiceOnset'] = trialEvents.loc[startIdx:nextStartIdx, 'Time'][rightButtonMask].mean()
                    trialStats.loc[idx, 'Outcome'] = 'correct button' if trialStats.loc[idx, 'Type'] == 1 else 'incorrect button'
                elif leftButtonMask.any():
                    trialStats.loc[idx, 'Choice'] = 'left'
                    trialStats.loc[idx, 'ChoiceOnset'] = trialEvents.loc[startIdx:nextStartIdx, 'Time'][leftButtonMask].mean()
                    trialStats.loc[idx, 'Outcome'] = 'correct button' if trialStats.loc[idx, 'Type'] == 0 else 'incorrect button'
                elif not rightButtonMask.any() and not leftButtonMask.any():
                    trialStats.loc[idx, 'Choice'] = 'none'
                    trialStats.loc[idx, 'Outcome'] = 'button timed out'
            except:
                print('Error detected!')
                #pdb.set_trace()
    return trialStats, trialEvents

#@profile
def plotTrialEvents(trialEvents, plotRange = None, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

    moveOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Movement Onset']
    moveOffIdx = trialEvents['Time'][trialEvents['Label'] == 'Movement Offset']
    RBOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Right Button Onset']
    LBOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Left Button Onset']
    RLOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Right LED Onset']
    LLOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Left LED Onset']

    if plotRange is not None:
        moveOnIdx = moveOnIdx[np.logical_and(moveOnIdx > plotRange[0], moveOnIdx < plotRange[1])]
        moveOffIdx = moveOffIdx[np.logical_and(moveOffIdx > plotRange[0], moveOffIdx < plotRange[1])]
        RBOnIdx = RBOnIdx[np.logical_and(RBOnIdx > plotRange[0], RBOnIdx < plotRange[1])]
        LBOnIdx = LBOnIdx[np.logical_and(LBOnIdx > plotRange[0], LBOnIdx < plotRange[1])]
        RLOnIdx = RLOnIdx[np.logical_and(RLOnIdx > plotRange[0], RLOnIdx < plotRange[1])]
        LLOnIdx = LLOnIdx[np.logical_and(LLOnIdx > plotRange[0], LLOnIdx < plotRange[1])]
    #pdb.set_trace()
    ax.plot(moveOnIdx, np.ones((len(moveOnIdx),1)), 'go')
    ax.plot(moveOffIdx, np.ones((len(moveOffIdx),1)), 'yo')
    ax.plot(RBOnIdx, np.ones((len(RBOnIdx),1)), 'cd')
    ax.plot(LBOnIdx, np.ones((len(LBOnIdx),1)), 'rd')
    ax.plot(RLOnIdx, np.ones((len(RLOnIdx),1)), 'bo')
    ax.plot(LLOnIdx, np.ones((len(LLOnIdx),1)), 'co')

#@profile
def plotMotor(motorData, plotRange = (0,-1), subset = None, addAxes = 0, collapse = False, ax = None):

    if subset is None:
        subset = motorData.columns

    if plotRange is None:
        xAxis = np.arange(len(motorData.index))
    else:
        xAxis = np.arange(int(plotRange[0]), int(plotRange[1]))
    #pdb.set_trace()
    if ax is not None:
        assert collapse == True
        fig = ax.figure
        ax = [ax]
    elif ax is None and not collapse:
        fig, ax = plt.subplots(nrows = len(subset) + addAxes, ncols = 1, sharex = True)
    else:
        fig, ax = plt.subplots(nrows = 1 + addAxes, ncols = 1, sharex = True)

    subsampleFactor = 30
    plotX = xAxis[::subsampleFactor] / 3e4
    for idx, column in enumerate(subset):
        #pdb.set_trace()
        #ax[idx].plot(xAxis, motorData.loc[slice(plotRange[0], plotRange[1]), column], label = column)
        if collapse:
            idx = 0
        ax[idx].plot(plotX, motorData.loc[xAxis[::subsampleFactor], column], label = column)
        ax[idx].legend()
    #pdb.set_trace()
    return fig, ax

def plotAverageAnalog(motorData, trialStats, alignTo, separateBy = None, subset = None, windowSize = (-0.25, 1), timeRange = None, showNow = False, ax = None, maxTrial = None, collapse = True, subsampleFactor = 30):
    if subset is None:
        subset = motorData.columns

    if separateBy is not None:
        uniqueCategories = pd.Series(trialStats.loc[:,separateBy].unique())
        uniqueCategories.dropna(inplace = True)

        if ax is None:
            fig, ax = plt.subplots(len(uniqueCategories),1)
        else:
            assert len(ax) == len(uniqueCategories)
            fig = ax.figure()

    else: # only one plot
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure


    if timeRange is not None:
        timeMask = np.logical_and(trialStats['FirstOnset'] > timeRange[0] * 3e4, trialStats['ChoiceOnset'] < timeRange[1] * 3e4)
        trialStats = trialStats.loc[timeMask, :]

    if maxTrial is not None:
        maxTrial = min(len(trialStats.index), maxTrial)
        trialStats = trialStats.iloc[:maxTrial, :]

    pdb.set_trace()
    motorDataSub = pd.DataFrame(motorData.iloc[::subsampleFactor, :].values, index = motorData.index[::subsampleFactor], columns = motorData.columns)
    timeWindow = list(range(int(windowSize[0] * 1e3), int(windowSize[1] * 1e3) + 1))
    colorPalette = sns.color_palette()
    for idx, column in enumerate(subset):
        #get the trace of interest
        trace = pd.DataFrame(index = trialStats.index, columns = timeWindow[:-1])

        for trialIdx, startTime in enumerate(trialStats[alignTo]):
            try:
                #pdb.set_trace()
                #print('Getting %s trace for trial %s' % (column, trialIdx))
                trace.iloc[trialIdx, :] = motorDataSub.loc[thisLocMask, column].values

                if maxTrial is not None:
                    if trialIdx >= maxTrial - 1:
                        break
            except:
                break
        #pdb.set_trace()
        if separateBy is not None:
            meanTrace = {category : pd.Series(index = timeWindow[:-1]) for category in uniqueCategories}
            errorTrace ={category : pd.Series(index = timeWindow[:-1]) for category in uniqueCategories}
            for category in uniqueCategories:
                    meanTrace[category] = trace.loc[trialStats[separateBy] == category].mean(axis = 0)
                    errorTrace[category] = trace.loc[trialStats[separateBy] == category].std(axis = 0)
        else:
            meanTrace = {'all' : trace.mean(axis = 0)}
            errorTrace = {'all' : trace.std(axis = 0)}

        for category, meanTraceThisCategory in meanTrace.items():
            if separateBy is not None:
                categoryIndex = pd.Index(uniqueCategories).get_loc(category)
                thisAx = ax[categoryIndex]
            else:
                thisAx = ax
            errorTraceThisCategory = errorTrace[category]
            thisAx.fill_between(timeWindow[:-1], meanTraceThisCategory-errorTraceThisCategory, meanTraceThisCategory+errorTraceThisCategory, alpha=0.4, facecolor=colorPalette[idx])
            thisAx.plot(timeWindow[:-1], meanTraceThisCategory, label=column, color=colorPalette[idx])
            thisAx.legend()
    #pdb.set_trace()
    return fig, ax

if __name__ == "__main__":
    ns5FilePath = 'D:/KiloSort/Trial001.ns5'
    inputIDs = {
        'A+' : 139,
        'A-' : 140,
        'B+' : 141,
        'B-' : 142,
        'Z+' : 143,
        'Z-' : 144,
        'leftLED' : 132,
        'leftBut' : 130,
        'rightLED' : 131,
        'rightBut' : 129,
        'simiTrigs' : 136,
        }

    """
    motorData = getMotorData(ns5FilePath, inputIDs, 0 , 'all')
    trials, trialEvents = getTrials(motorData)
    """
    motorData = getMotorData(ns5FilePath, inputIDs, 0 , 'all')
    trialStats  = pd.read_pickle('D:/Staging/Trial001_trialStats.pickle')
    trialEvents = pd.read_pickle('D:/Staging/Trial001_trialEvents.pickle')
    timeRange = 30
    #pdb.set_trace()

    while True:
        plotAxes = plotMotor(motorData, plotRange = ((timeRange - 30)* 30e3, timeRange * 30e3), subset = ['position', 'leftLED_int', 'rightLED_int', 'leftBut_int', 'rightBut_int','A_int'])
        plotTrialEvents(trialEvents, plotRange = ((timeRange - 30)* 30e3, timeRange * 30e3), ax = plotAxes[-1])
        plt.show(block = True)
        timeRange = timeRange + 30
    #

    #
