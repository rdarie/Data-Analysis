import pdb
from dataAnalysis.helperFunctions.helper_functions import *
import pandas as pd

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt

from sklearn.mixture import GaussianMixture

import seaborn as sns
from scipy.signal import hilbert

def debounceLine(dataLine):
    pass

def getTransitionIdx(motorData):

    Adiff = motorData['A_int'].diff().abs()
    Bdiff = motorData['B_int'].diff().abs()

    transitionMask = np.logical_or(Adiff == 1, Bdiff == 1)
    transitionIdx = motorData.index[transitionMask].tolist()

    return transitionMask, transitionIdx

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

    transitionMask, transitionIdx = getTransitionIdx(motorData)
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

    transitionMask, transitionIdx = getTransitionIdx(motorData)
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
    skipAhead = 0
    skipAheadInc = moveThreshold

    firstIdx = motorData.index[0]
    lastIdx = motorData.index[-1]
    lookForStarts = True
    for idx in transitionIdx:
        #check that we're not out of bounds])
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
            if not skipAhead:
                transitionInPast = sum(transitionMask[idx - windowLen : idx])
                transitionInFuture = sum(transitionMask[idx : idx + windowLen])
                if transitionInPast < noMoveThreshold and transitionInFuture > moveThreshold and lookForStarts:
                    movementOnsetIdx.append(idx)
                    trialEvents = trialEvents.append({'Time':idx,'Label':'Movement Onset'}, ignore_index = True)
                    skipAhead = skipAheadInc
                    lookForStarts = False
                if transitionInPast > moveThreshold and transitionInFuture < noMoveThreshold and not lookForStarts:
                    movementOffsetIdx.append(idx)
                    trialEvents = trialEvents.append({'Time':idx,'Label':'Movement Offset'}, ignore_index = True)
                    lookForStarts = True
            else:
                skipAhead = skipAhead - 1

    if len(movementOnsetIdx) > len(movementOffsetIdx):
        movementOnsetIdx = movementOnsetIdx[:-1]

    assert len(movementOnsetIdx) == len(movementOffsetIdx)

    # detection of right button choice (debouce)
    skipAhead = 0
    # look at this size window and determine if there's an index there
    windowLen = int(200e-3 / 3e-4)
    # How many transitions should be in the window in order for this to count as a "start"
    onThreshold = 20
    # at most how many transitions can be in the idx for us not to count it?
    offThreshold = 2
    for idx in motorData.index[rightButEdges].tolist():
        #check that we're not out of bounds])
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
            if not skipAhead:
                transitionInPast = sum(motorData['rightBut_int'][idx - windowLen : idx])
                transitionInFuture = sum(motorData['rightBut_int'][idx : idx + windowLen])
                if transitionInPast < offThreshold and transitionInFuture > onThreshold:
                    rightButOnsetIdx.append(idx)
                    trialEvents = trialEvents.append({'Time':idx,'Label':'Right Button Onset'}, ignore_index = True)
                    skipAhead = skipAheadInc
            else:
                skipAhead = skipAhead - 1

    skipAhead = 0
    for idx in motorData.index[leftButEdges].tolist():
        #check that we're not out of bounds])
        #if idx > 161400 and idx < 161600:
        #    pdb.set_trace()
        if idx - windowLen >= firstIdx and idx + windowLen <= lastIdx:
            if not skipAhead:
                transitionInPast = sum(motorData['leftBut_int'][idx - windowLen : idx])
                transitionInFuture = sum(motorData['leftBut_int'][idx : idx + windowLen])
                if transitionInPast < offThreshold and transitionInFuture > onThreshold:
                    leftButOnsetIdx.append(idx)
                    trialEvents = trialEvents.append({'Time':idx,'Label':'Left Button Onset'}, ignore_index = True)
                    skipAhead = skipAheadInc
            else:
                skipAhead = skipAhead - 1

    for idx in rightLEDOnsetIdx:
        trialEvents = trialEvents.append({'Time':idx,'Label':'Right LED Onset'}, ignore_index = True)
    for idx in leftLEDOnsetIdx:
        trialEvents = trialEvents.append({'Time':idx,'Label':'Left LED Onset'}, ignore_index = True)

    #pdb.set_trace()
    trialEvents.sort_values('Time', inplace = True)
    if trialType == '2AFC':
        trials = pd.DataFrame()

    return trials, trialEvents

def plotTrialEvents(trialEvents):
    moveOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Movement Onset']
    moveOffIdx = trialEvents['Time'][trialEvents['Label'] == 'Movement Offset']
    RBOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Right Button Onset']
    LBOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Left Button Onset']
    RLOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Right LED Onset']
    LLOnIdx = trialEvents['Time'][trialEvents['Label'] == 'Left LED Onset']
    #pdb.set_trace()
    plt.plot(moveOnIdx, np.ones((len(moveOnIdx),1)), 'go')
    plt.plot(moveOffIdx, np.ones((len(moveOffIdx),1)), 'yo')
    plt.plot(RBOnIdx, np.ones((len(RBOnIdx),1)), 'cd')
    plt.plot(LBOnIdx, np.ones((len(LBOnIdx),1)), 'rd')
    plt.plot(RLOnIdx, np.ones((len(RLOnIdx),1)), 'bo')
    plt.plot(LLOnIdx, np.ones((len(LLOnIdx),1)), 'co')

def plotMotor(motorData, plotRange = (0,-1), subset = None):
    if subset is None:
        subset = motorData.columns
    fig, ax = plt.subplots(nrows = len(subset), ncols = 1, sharex = True)
    for idx, column in enumerate(subset):
        ax[idx].plot(motorData.loc[slice(plotRange[0], plotRange[1]), column], label = column)
        ax[idx].legend()

#if __name__ == "__main__":
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

motorData = getMotorData(ns5FilePath, inputIDs, 10 , 30)
trials, trialEvents = getTrials(motorData)
plotMotor(motorData, plotRange = (0e4, 200e4), subset = ['position', 'leftLED_int', 'rightLED_int', 'leftBut_int', 'rightBut_int','A_int'])
plotTrialEvents(trialEvents)
#

plt.show(block = False)
pdb.set_trace()
