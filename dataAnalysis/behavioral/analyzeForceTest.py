import pickle, sys, math, pdb, h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from brPY.brpylib import NsxFile, NevFile, brpylib_ver
from dataAnalysis.helperFunctions.helper_functions import getNSxData
dataDir = 'W:/ENG_Neuromotion_Shared/group/Monkey_Neural_Recordings/Starbuck_Bilateral_Recordings/201802191132-Starbuck-Freely_Moving_Force_Plates'
pythonDataName = '/Force Plate Data/201802191105-Freely_Moving_Force_Plates_Starbuck002.p'
simiTriggersPath = 'E:/temp/201802191132-Starbuck-Freely_Moving_Force_Plates-Simi_Triggers-Trial002.h5'
nevDataName = '/201802191132-Starbuck-Freely_Moving_Force_Plates-Array1480_Right_Side-Trial002.mat'

sns.set(font_scale=1.5)
#Load calibration data
calibrationFiles = [0,1]
calibrationFiles[1] = 'C:/Users/Radu/Documents/GitHub/Data-Analysis/D1550_1.acl'
calibrationFiles[0] = 'C:/Users/Radu/Documents/GitHub/Data-Analysis/D1551_2.acl'

def readForceData(nevDataName, calibrationFiles):
    """
    Reads a .mat file derived from a NEV file and parses the force data inside
    """
    #read in calibration files
    S = [0,1]
    S[1] = pd.read_csv(calibrationFiles[0], names = range(1,13), index_col = False, header = None).reindex(index = range(1,7))
    S[0] = pd.read_csv(calibrationFiles[1], names = range(1,13), index_col = False).reindex(index = range(1,7))
    #
    # Open nev data
    #
    f = h5py.File(dataDir + nevDataName,'r')

    timeStampsHDF = f.get('NEV/Data/SerialDigitalIO/TimeStampSec')
    timeStamps = np.array(timeStampsHDF)
    eventsHDF = f.get('NEV/Data/SerialDigitalIO/UnparsedData')
    events = np.array(eventsHDF)
    reasonHDF = f.get('NEV/Data/SerialDigitalIO/InsertionReason')
    reason = np.array(reasonHDF)

    nevData = {
        'dig_events': {
            'Data' : events,
            'TimeStamps' : timeStamps,
            'Reason' : reason
            }
        }

    nPoints = 3
    # 24 * 16 bit blocks per reading
    addressIdx = range(1,48*nPoints,2)
    matchStr = ['f123456789abba9876543210', 'f123456789abba987654321f', '0123456789abba987654321f', '0123456789abba9876543210']
    idNibbles = '' # will hold bits 4-7 of the high byte for each number assuming the reading frame starts with the first byte

    for idx in addressIdx:
        # the top 8 bits of each 16 bit number is the high byte and it contains the counter (unless the reading frame starts with two, we handle that case below)
        highByte = format(nevData['dig_events']['Data'][0][idx], '02x')
        print(highByte)
        # make a list of the nibbles that should contain the counter
        idNibbles = idNibbles + highByte[0]
        print(idNibbles)

    addressStartID = None
    # scan through the nibble list to find something that looks like the counter
    findMatches = [combo in idNibbles for combo in matchStr]
    if any(findMatches):
        for combo in matchStr:
            addressStartID = idNibbles.find(combo)
            if addressStartID != -1:
                break
    else:
        addressIdx = range(2,48*nPoints,2)
        idNibbles = ''  # will hold bits 4-7 of the high byte for each number assuming the reading frame starts with the second byte

        for idx in addressIdx:
            highByte = format(nevData['dig_events']['Data'][0][idx], '02x')
            #print(highByte)
            idNibbles = idNibbles + highByte[0]

        addressStartID = None
        findMatches = [combo in idNibbles for combo in matchStr]
        if any(findMatches):
            for combo in matchStr:
                addressStartID = idNibbles.find(combo)
                if addressStartID != -1:
                    break

    assert addressStartID != -1

    startIdx = addressIdx[addressStartID] - 1
    #format(nevData['dig_events']['Data'][0][startIdx], '02x')

    addressIdxRange = range(1, 48, 2)
    channelNames = ['Ax0', 'Ay0', 'Az0', 'Bx0', 'By0', 'Bz0', 'Cx0', 'Cy0', 'Cz0', 'Dx0', 'Dy0', 'Dz0',
        'Ax1', 'Ay1', 'Az1', 'Bx1', 'By1', 'Bz1', 'Cx1', 'Cy1', 'Cz1', 'Dx1', 'Dy1', 'Dz1']
    # Sanity check: sensor readings must be coming in in this order:
    # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    expectedSensorIdxOrder = list(range(12)) + list(range(12))[::-1]

    channelLookup = {i : channelNames[count] for count, i in enumerate(addressIdxRange)}

    # if there's junk at the end, discount the number of readings
    nReadings = math.floor(len(nevData['dig_events']['Data'][0]) / 48) - math.floor(startIdx / 48)
    #nReadings = 12500
    blockIdxRange = range(0,48*nReadings,48) # 24 * 16 bit or 48 byte long blocks per reading, the index of each block start
    NevProcData = {name : [] for name in channelNames + ['Sync']}

    #make an np.array that will hold the times at which the first block of each reading was read
    tNevSerial = []
    # Collect nReadings worth of data:

    #should we give up trying to pad missing bytes?
    giveUp = False
    # find addresses in future bytes to check for possibly corrupt data
    def forwardAddresses(nevData, lookAhead):
        ret = [ format(i, '02x')[0] for i in nevData[lookAhead] ]
        ret = [ i if i != 'f' else '0' for i in ret]
        return ''.join(ret)

    def isValidAddressSequence(sequence):
        return sequence in '0123456789abba98765432100123456789abba9876543210'

    for readingIdx, blockIdx in enumerate(blockIdxRange):
        #print('Reading index was: ')
        #print(readingIdx)
        #print('Reading index was: ')
        #print(blockIdx)
        # must perform sanity check that we are reading all 48 bits
        sensorIds = ''
        for byteIdx, addressIdx in enumerate(addressIdxRange):
            #print('byteIdx index was: ')
            #print(byteIdx)
            #print('addressIdx index was: ')
            #print(addressIdx)

            lowByte = nevData['dig_events']['Data'][0][blockIdx + addressIdx - 1 + startIdx]
            highByte = nevData['dig_events']['Data'][0][blockIdx + addressIdx + startIdx]
            lowByteStr = format(lowByte, '02x')
            highByteStr = format(highByte, '02x')
            #if readingIdx == 0:
            #    pdb.set_trace()
            if addressIdx == 1:
                #pdb.set_trace()
                currTimeStamp = nevData['dig_events']['TimeStamps'][blockIdx + addressIdx - 1 + startIdx]
                if type(currTimeStamp) == np.ndarray:
                    currTimeStamp = currTimeStamp[0]
                #print(type(currTimeStamp))
                if currTimeStamp == 0:
                    break

                syncReading = 0 if highByteStr[0] == 'f' else 1
                NevProcData['Sync'].append(syncReading)
                tNevSerial.append(currTimeStamp)

            #print('address index is %d' % addressIdx)
            sensorIndex = int('0x' + highByteStr[0], 0)
            sensorIndex = 0 if sensorIndex == 15 else sensorIndex # roll f's into 0's
            sensorIds = sensorIds + highByteStr[0]
            #print('sensor index is %d' % highByteIndex)

            if expectedSensorIdxOrder[byteIdx] != sensorIndex and not giveUp:
                # We somehow skipped a reading! This is really bad because it messes up the reading frame and any subsequent data.....
                currIndexIntoNevData = blockIdx + addressIdx + startIdx
                lookAhead = [currIndexIntoNevData + i for i in range(-2, 48, 2)]
                print('Reading frame error at %d' % (currIndexIntoNevData))
                forwardPeek = forwardAddresses(nevData['dig_events']['Data'][0], lookAhead)
                print(' Forward look is : ' + forwardPeek)

                #if all zeros, we're at the end, give up
                if '0' * (len(forwardPeek) - 1) == forwardPeek[1:]:
                    giveUp = True

                while not isValidAddressSequence( forwardPeek ) and not giveUp:
                    #pdb.set_trace()
                    newAddressByte = int ( '0x' + format( expectedSensorIdxOrder[byteIdx], '01x') + '0', 0)
                    nevData['dig_events']['Data'] = np.array([np.insert(nevData['dig_events']['Data'][0], currIndexIntoNevData, newAddressByte )])
                    nevData['dig_events']['TimeStamps'] = np.array(np.insert(nevData['dig_events']['TimeStamps'], currIndexIntoNevData, 0 ))
                    forwardPeek = forwardAddresses(nevData['dig_events']['Data'][0], lookAhead)
                    print(forwardPeek)

                NevProcData[channelLookup[addressIdx]].append(0)
                sensorIds = sensorIds[:-1] + format(expectedSensorIdxOrder[byteIdx], '01x')
                #
            #
            else:
                NevProcData[channelLookup[addressIdx]].append(int(highByteStr[1] + lowByteStr[0] + lowByteStr[1], 16))

        else:
            continue  # executed if the loop ended normally (no break)
        break  # executed if 'continue' was skipped (break)

            #NevProcData[channelLookup[addressIdx]][readingIdx] = int(highByteStr[1] + lowByteStr[0] + lowByteStr[1], 16)

        #print('Sensor ids were: ' + sensorIds)
        #if sensorIds not in matchStr:
        #    pdb.set_trace()
    """
    Zero the measurements
    """
    #plt.plot(np.diff(nevData['dig_events']['TimeStamps'][0]))
    #plt.show()
    for key in NevProcData:
        if key != 'Sync':
            NevProcData[key] = np.asarray(NevProcData[key]) - int(round(np.mean(NevProcData[key][:1000])))

    ProcData = {
        'Fx0' : ( (NevProcData['Ax0'] * S[0].loc[1,1]) + (NevProcData['Ay0'] * S[0].loc[1,2]) + (NevProcData['Az0'] * S[0].loc[1,3]) + (NevProcData['Bx0'] * S[0].loc[1,4])+ (NevProcData['By0'] * S[0].loc[1,5])+ (NevProcData['Bz0'] * S[0].loc[1,6]) + (NevProcData['Cx0'] * S[0].loc[1,7]) + (NevProcData['Cy0'] * S[0].loc[1,8]) + (NevProcData['Cz0'] * S[0].loc[1,9]) + (NevProcData['Dx0'] * S[0].loc[1,10])+ (NevProcData['Dy0'] * S[0].loc[1,11]) + (NevProcData['Dz0'] * S[0].loc[1,12]) ) * 453.592,#lb to g
        'Fy0' : ( (NevProcData['Ax0'] * S[0].loc[2,1]) + (NevProcData['Ay0'] * S[0].loc[2,2]) + (NevProcData['Az0'] * S[0].loc[2,3]) + (NevProcData['Bx0'] * S[0].loc[2,4])+ (NevProcData['By0'] * S[0].loc[2,5])+ (NevProcData['Bz0'] * S[0].loc[2,6]) + (NevProcData['Cx0'] * S[0].loc[2,7]) + (NevProcData['Cy0'] * S[0].loc[2,8]) + (NevProcData['Cz0'] * S[0].loc[2,9]) + (NevProcData['Dx0'] * S[0].loc[2,10])+ (NevProcData['Dy0'] * S[0].loc[2,11]) + (NevProcData['Dz0'] * S[0].loc[2,12]) ) * 453.592,#lb to g
        'Fz0' : ( (NevProcData['Ax0'] * S[0].loc[3,1]) + (NevProcData['Ay0'] * S[0].loc[3,2]) + (NevProcData['Az0'] * S[0].loc[3,3]) + (NevProcData['Bx0'] * S[0].loc[3,4])+ (NevProcData['By0'] * S[0].loc[3,5])+ (NevProcData['Bz0'] * S[0].loc[3,6]) + (NevProcData['Cx0'] * S[0].loc[3,7]) + (NevProcData['Cy0'] * S[0].loc[3,8]) + (NevProcData['Cz0'] * S[0].loc[3,9]) + (NevProcData['Dx0'] * S[0].loc[3,10])+ (NevProcData['Dy0'] * S[0].loc[3,11]) + (NevProcData['Dz0'] * S[0].loc[3,12]) ) * 453.592,#lb to g
        'Mx0' : ( (NevProcData['Ax0'] * S[0].loc[4,1]) + (NevProcData['Ay0'] * S[0].loc[4,2]) + (NevProcData['Az0'] * S[0].loc[4,3]) + (NevProcData['Bx0'] * S[0].loc[4,4])+ (NevProcData['By0'] * S[0].loc[4,5])+ (NevProcData['Bz0'] * S[0].loc[4,6]) + (NevProcData['Cx0'] * S[0].loc[4,7]) + (NevProcData['Cy0'] * S[0].loc[4,8]) + (NevProcData['Cz0'] * S[0].loc[4,9]) + (NevProcData['Dx0'] * S[0].loc[4,10])+ (NevProcData['Dy0'] * S[0].loc[4,11]) + (NevProcData['Dz0'] * S[0].loc[4,12]) ) * 0.113, #Ino N*m
        'My0' : ( (NevProcData['Ax0'] * S[0].loc[5,1]) + (NevProcData['Ay0'] * S[0].loc[5,2]) + (NevProcData['Az0'] * S[0].loc[5,3]) + (NevProcData['Bx0'] * S[0].loc[5,4])+ (NevProcData['By0'] * S[0].loc[5,5])+ (NevProcData['Bz0'] * S[0].loc[5,6]) + (NevProcData['Cx0'] * S[0].loc[5,7]) + (NevProcData['Cy0'] * S[0].loc[5,8]) + (NevProcData['Cz0'] * S[0].loc[5,9]) + (NevProcData['Dx0'] * S[0].loc[5,10])+ (NevProcData['Dy0'] * S[0].loc[5,11]) + (NevProcData['Dz0'] * S[0].loc[5,12]) ) * 0.113, #Ino N*m
        'Mz0' : ( (NevProcData['Ax0'] * S[0].loc[6,1]) + (NevProcData['Ay0'] * S[0].loc[6,2]) + (NevProcData['Az0'] * S[0].loc[6,3]) + (NevProcData['Bx0'] * S[0].loc[6,4])+ (NevProcData['By0'] * S[0].loc[6,5])+ (NevProcData['Bz0'] * S[0].loc[6,6]) + (NevProcData['Cx0'] * S[0].loc[6,7]) + (NevProcData['Cy0'] * S[0].loc[6,8]) + (NevProcData['Cz0'] * S[0].loc[6,9]) + (NevProcData['Dx0'] * S[0].loc[6,10])+ (NevProcData['Dy0'] * S[0].loc[6,11]) + (NevProcData['Dz0'] * S[0].loc[6,12]) ) * 0.113, #Ino N*m
        'Fx1' : ( (NevProcData['Ax1'] * S[1].loc[1,1]) + (NevProcData['Ay1'] * S[1].loc[1,2]) + (NevProcData['Az1'] * S[1].loc[1,3]) + (NevProcData['Bx1'] * S[1].loc[1,4])+ (NevProcData['By1'] * S[1].loc[1,5])+ (NevProcData['Bz1'] * S[1].loc[1,6]) + (NevProcData['Cx1'] * S[1].loc[1,7]) + (NevProcData['Cy1'] * S[1].loc[1,8]) + (NevProcData['Cz1'] * S[1].loc[1,9]) + (NevProcData['Dx1'] * S[1].loc[1,10])+ (NevProcData['Dy1'] * S[1].loc[1,11]) + (NevProcData['Dz1'] * S[1].loc[1,12]) ) * 453.592,#lb to g
        'Fy1' : ( (NevProcData['Ax1'] * S[1].loc[2,1]) + (NevProcData['Ay1'] * S[1].loc[2,2]) + (NevProcData['Az1'] * S[1].loc[2,3]) + (NevProcData['Bx1'] * S[1].loc[2,4])+ (NevProcData['By1'] * S[1].loc[2,5])+ (NevProcData['Bz1'] * S[1].loc[2,6]) + (NevProcData['Cx1'] * S[1].loc[2,7]) + (NevProcData['Cy1'] * S[1].loc[2,8]) + (NevProcData['Cz1'] * S[1].loc[2,9]) + (NevProcData['Dx1'] * S[1].loc[2,10])+ (NevProcData['Dy1'] * S[1].loc[2,11]) + (NevProcData['Dz1'] * S[1].loc[2,12]) ) * 453.592,#lb to g
        'Fz1' : ( (NevProcData['Ax1'] * S[1].loc[3,1]) + (NevProcData['Ay1'] * S[1].loc[3,2]) + (NevProcData['Az1'] * S[1].loc[3,3]) + (NevProcData['Bx1'] * S[1].loc[3,4])+ (NevProcData['By1'] * S[1].loc[3,5])+ (NevProcData['Bz1'] * S[1].loc[3,6]) + (NevProcData['Cx1'] * S[1].loc[3,7]) + (NevProcData['Cy1'] * S[1].loc[3,8]) + (NevProcData['Cz1'] * S[1].loc[3,9]) + (NevProcData['Dx1'] * S[1].loc[3,10])+ (NevProcData['Dy1'] * S[1].loc[3,11]) + (NevProcData['Dz1'] * S[1].loc[3,12]) ) * 453.592,#lb to g
        'Mx1' : ( (NevProcData['Ax1'] * S[1].loc[4,1]) + (NevProcData['Ay1'] * S[1].loc[4,2]) + (NevProcData['Az1'] * S[1].loc[4,3]) + (NevProcData['Bx1'] * S[1].loc[4,4])+ (NevProcData['By1'] * S[1].loc[4,5])+ (NevProcData['Bz1'] * S[1].loc[4,6]) + (NevProcData['Cx1'] * S[1].loc[4,7]) + (NevProcData['Cy1'] * S[1].loc[4,8]) + (NevProcData['Cz1'] * S[1].loc[4,9]) + (NevProcData['Dx1'] * S[1].loc[4,10])+ (NevProcData['Dy1'] * S[1].loc[4,11]) + (NevProcData['Dz1'] * S[1].loc[4,12]) ) * 0.113, #Ino N*m
        'My1' : ( (NevProcData['Ax1'] * S[1].loc[5,1]) + (NevProcData['Ay1'] * S[1].loc[5,2]) + (NevProcData['Az1'] * S[1].loc[5,3]) + (NevProcData['Bx1'] * S[1].loc[5,4])+ (NevProcData['By1'] * S[1].loc[5,5])+ (NevProcData['Bz1'] * S[1].loc[5,6]) + (NevProcData['Cx1'] * S[1].loc[5,7]) + (NevProcData['Cy1'] * S[1].loc[5,8]) + (NevProcData['Cz1'] * S[1].loc[5,9]) + (NevProcData['Dx1'] * S[1].loc[5,10])+ (NevProcData['Dy1'] * S[1].loc[5,11]) + (NevProcData['Dz1'] * S[1].loc[5,12]) ) * 0.113, #Ino N*m
        'Mz1' : ( (NevProcData['Ax1'] * S[1].loc[6,1]) + (NevProcData['Ay1'] * S[1].loc[6,2]) + (NevProcData['Az1'] * S[1].loc[6,3]) + (NevProcData['Bx1'] * S[1].loc[6,4])+ (NevProcData['By1'] * S[1].loc[6,5])+ (NevProcData['Bz1'] * S[1].loc[6,6]) + (NevProcData['Cx1'] * S[1].loc[6,7]) + (NevProcData['Cy1'] * S[1].loc[6,8]) + (NevProcData['Cz1'] * S[1].loc[6,9]) + (NevProcData['Dx1'] * S[1].loc[6,10])+ (NevProcData['Dy1'] * S[1].loc[6,11]) + (NevProcData['Dz1'] * S[1].loc[6,12]) ) * 0.113, #Ino N*m
        'Sync': NevProcData['Sync'],
        't' : np.asarray(tNevSerial)
        }

    return ProcData

ProcData = readForceData(nevDataName, calibrationFiles)

plt.plot(ProcData['t'], ProcData['Fz0'], label = 'Force Plate 0', marker = 'o')
plt.plot(ProcData['t'], ProcData['Fz1'], label = 'Force Plate 1', marker = 'o')

# open data saved via python
def readPythonForceData(pythonDataName, calibrationFiles):
    #read in calibration files
    S = [0,1]
    S[1] = pd.read_csv(calibrationFiles[0], names = range(1,13), index_col = False, header = None).reindex(index = range(1,7))
    S[0] = pd.read_csv(calibrationFiles[1], names = range(1,13), index_col = False).reindex(index = range(1,7))

    pythonData = pickle.load(open(dataDir + pythonDataName, 'rb'))
    nReadings = len(pythonData)

    addressIdxRange = range(1, 48, 2)
    channelNames = ['Ax0', 'Ay0', 'Az0', 'Bx0', 'By0', 'Bz0', 'Cx0', 'Cy0', 'Cz0', 'Dx0', 'Dy0', 'Dz0',
        'Ax1', 'Ay1', 'Az1', 'Bx1', 'By1', 'Bz1', 'Cx1', 'Cy1', 'Cz1', 'Dx1', 'Dy1', 'Dz1']
    channelLookup = {i : channelNames[count] for count, i in enumerate(addressIdxRange)}

    procPythonData = {name : [0 for i in range(nReadings)] for name in channelNames + ['Sync']}
    addressIdxRange = range(1,48,2)

    for readingIdx, datum in enumerate(pythonData[:nReadings]):
        for addressIdx in addressIdxRange:
            lowByte = datum[addressIdx - 1]
            highByte = datum[addressIdx]
            lowByteStr = format(lowByte, '02x')
            highByteStr = format(highByte, '02x')
            #print(lowByteStr + highByteStr)
            procPythonData[channelLookup[addressIdx]][readingIdx] = int(highByteStr[1] + lowByteStr[0] + lowByteStr[1], 16)
            if addressIdx == 1:
                #pdb.set_trace()
                procPythonData['Sync'][readingIdx] = 0 if highByteStr[0] == 'f' else 1

    for key in procPythonData:
        if key != 'Sync':
            procPythonData[key] = np.asarray(procPythonData[key]) - int(round(np.mean(procPythonData[key][:1000])))

    calibPythonData = {
        'Fx0' : ( (procPythonData['Ax0'] * S[0].loc[1,1]) + (procPythonData['Ay0'] * S[0].loc[1,2]) + (procPythonData['Az0'] * S[0].loc[1,3]) + (procPythonData['Bx0'] * S[0].loc[1,4])+ (procPythonData['By0'] * S[0].loc[1,5])+ (procPythonData['Bz0'] * S[0].loc[1,6]) + (procPythonData['Cx0'] * S[0].loc[1,7]) + (procPythonData['Cy0'] * S[0].loc[1,8]) + (procPythonData['Cz0'] * S[0].loc[1,9]) + (procPythonData['Dx0'] * S[0].loc[1,10])+ (procPythonData['Dy0'] * S[0].loc[1,11]) + (procPythonData['Dz0'] * S[0].loc[1,12]) ) * 453.592, #lb to g
        'Fy0' : ( (procPythonData['Ax0'] * S[0].loc[2,1]) + (procPythonData['Ay0'] * S[0].loc[2,2]) + (procPythonData['Az0'] * S[0].loc[2,3]) + (procPythonData['Bx0'] * S[0].loc[2,4])+ (procPythonData['By0'] * S[0].loc[2,5])+ (procPythonData['Bz0'] * S[0].loc[2,6]) + (procPythonData['Cx0'] * S[0].loc[2,7]) + (procPythonData['Cy0'] * S[0].loc[2,8]) + (procPythonData['Cz0'] * S[0].loc[2,9]) + (procPythonData['Dx0'] * S[0].loc[2,10])+ (procPythonData['Dy0'] * S[0].loc[2,11]) + (procPythonData['Dz0'] * S[0].loc[2,12]) ) * 453.592, #lb to g
        'Fz0' : ( (procPythonData['Ax0'] * S[0].loc[3,1]) + (procPythonData['Ay0'] * S[0].loc[3,2]) + (procPythonData['Az0'] * S[0].loc[3,3]) + (procPythonData['Bx0'] * S[0].loc[3,4])+ (procPythonData['By0'] * S[0].loc[3,5])+ (procPythonData['Bz0'] * S[0].loc[3,6]) + (procPythonData['Cx0'] * S[0].loc[3,7]) + (procPythonData['Cy0'] * S[0].loc[3,8]) + (procPythonData['Cz0'] * S[0].loc[3,9]) + (procPythonData['Dx0'] * S[0].loc[3,10])+ (procPythonData['Dy0'] * S[0].loc[3,11]) + (procPythonData['Dz0'] * S[0].loc[3,12]) ) * 453.592, #lb to g
        'Mx0' : ( (procPythonData['Ax0'] * S[0].loc[4,1]) + (procPythonData['Ay0'] * S[0].loc[4,2]) + (procPythonData['Az0'] * S[0].loc[4,3]) + (procPythonData['Bx0'] * S[0].loc[4,4])+ (procPythonData['By0'] * S[0].loc[4,5])+ (procPythonData['Bz0'] * S[0].loc[4,6]) + (procPythonData['Cx0'] * S[0].loc[4,7]) + (procPythonData['Cy0'] * S[0].loc[4,8]) + (procPythonData['Cz0'] * S[0].loc[4,9]) + (procPythonData['Dx0'] * S[0].loc[4,10])+ (procPythonData['Dy0'] * S[0].loc[4,11]) + (procPythonData['Dz0'] * S[0].loc[4,12]) ) * 0.113, #Ino N*m
        'My0' : ( (procPythonData['Ax0'] * S[0].loc[5,1]) + (procPythonData['Ay0'] * S[0].loc[5,2]) + (procPythonData['Az0'] * S[0].loc[5,3]) + (procPythonData['Bx0'] * S[0].loc[5,4])+ (procPythonData['By0'] * S[0].loc[5,5])+ (procPythonData['Bz0'] * S[0].loc[5,6]) + (procPythonData['Cx0'] * S[0].loc[5,7]) + (procPythonData['Cy0'] * S[0].loc[5,8]) + (procPythonData['Cz0'] * S[0].loc[5,9]) + (procPythonData['Dx0'] * S[0].loc[5,10])+ (procPythonData['Dy0'] * S[0].loc[5,11]) + (procPythonData['Dz0'] * S[0].loc[5,12]) ) * 0.113, #Ino N*m
        'Mz0' : ( (procPythonData['Ax0'] * S[0].loc[6,1]) + (procPythonData['Ay0'] * S[0].loc[6,2]) + (procPythonData['Az0'] * S[0].loc[6,3]) + (procPythonData['Bx0'] * S[0].loc[6,4])+ (procPythonData['By0'] * S[0].loc[6,5])+ (procPythonData['Bz0'] * S[0].loc[6,6]) + (procPythonData['Cx0'] * S[0].loc[6,7]) + (procPythonData['Cy0'] * S[0].loc[6,8]) + (procPythonData['Cz0'] * S[0].loc[6,9]) + (procPythonData['Dx0'] * S[0].loc[6,10])+ (procPythonData['Dy0'] * S[0].loc[6,11]) + (procPythonData['Dz0'] * S[0].loc[6,12]) ) * 0.113, #Ino N*m
        'Fx1' : ( (procPythonData['Ax1'] * S[1].loc[1,1]) + (procPythonData['Ay1'] * S[1].loc[1,2]) + (procPythonData['Az1'] * S[1].loc[1,3]) + (procPythonData['Bx1'] * S[1].loc[1,4])+ (procPythonData['By1'] * S[1].loc[1,5])+ (procPythonData['Bz1'] * S[1].loc[1,6]) + (procPythonData['Cx1'] * S[1].loc[1,7]) + (procPythonData['Cy1'] * S[1].loc[1,8]) + (procPythonData['Cz1'] * S[1].loc[1,9]) + (procPythonData['Dx1'] * S[1].loc[1,10])+ (procPythonData['Dy1'] * S[1].loc[1,11]) + (procPythonData['Dz1'] * S[1].loc[1,12]) ) * 453.592,#lb to g
        'Fy1' : ( (procPythonData['Ax1'] * S[1].loc[2,1]) + (procPythonData['Ay1'] * S[1].loc[2,2]) + (procPythonData['Az1'] * S[1].loc[2,3]) + (procPythonData['Bx1'] * S[1].loc[2,4])+ (procPythonData['By1'] * S[1].loc[2,5])+ (procPythonData['Bz1'] * S[1].loc[2,6]) + (procPythonData['Cx1'] * S[1].loc[2,7]) + (procPythonData['Cy1'] * S[1].loc[2,8]) + (procPythonData['Cz1'] * S[1].loc[2,9]) + (procPythonData['Dx1'] * S[1].loc[2,10])+ (procPythonData['Dy1'] * S[1].loc[2,11]) + (procPythonData['Dz1'] * S[1].loc[2,12]) ) * 453.592, #lb to g
        'Fz1' : ( (procPythonData['Ax1'] * S[1].loc[3,1]) + (procPythonData['Ay1'] * S[1].loc[3,2]) + (procPythonData['Az1'] * S[1].loc[3,3]) + (procPythonData['Bx1'] * S[1].loc[3,4])+ (procPythonData['By1'] * S[1].loc[3,5])+ (procPythonData['Bz1'] * S[1].loc[3,6]) + (procPythonData['Cx1'] * S[1].loc[3,7]) + (procPythonData['Cy1'] * S[1].loc[3,8]) + (procPythonData['Cz1'] * S[1].loc[3,9]) + (procPythonData['Dx1'] * S[1].loc[3,10])+ (procPythonData['Dy1'] * S[1].loc[3,11]) + (procPythonData['Dz1'] * S[1].loc[3,12]) ) * 453.592, #lb to g
        'Mx1' : ( (procPythonData['Ax1'] * S[1].loc[4,1]) + (procPythonData['Ay1'] * S[1].loc[4,2]) + (procPythonData['Az1'] * S[1].loc[4,3]) + (procPythonData['Bx1'] * S[1].loc[4,4])+ (procPythonData['By1'] * S[1].loc[4,5])+ (procPythonData['Bz1'] * S[1].loc[4,6]) + (procPythonData['Cx1'] * S[1].loc[4,7]) + (procPythonData['Cy1'] * S[1].loc[4,8]) + (procPythonData['Cz1'] * S[1].loc[4,9]) + (procPythonData['Dx1'] * S[1].loc[4,10])+ (procPythonData['Dy1'] * S[1].loc[4,11]) + (procPythonData['Dz1'] * S[1].loc[4,12]) ) * 0.113, #Ino N*m
        'My1' : ( (procPythonData['Ax1'] * S[1].loc[5,1]) + (procPythonData['Ay1'] * S[1].loc[5,2]) + (procPythonData['Az1'] * S[1].loc[5,3]) + (procPythonData['Bx1'] * S[1].loc[5,4])+ (procPythonData['By1'] * S[1].loc[5,5])+ (procPythonData['Bz1'] * S[1].loc[5,6]) + (procPythonData['Cx1'] * S[1].loc[5,7]) + (procPythonData['Cy1'] * S[1].loc[5,8]) + (procPythonData['Cz1'] * S[1].loc[5,9]) + (procPythonData['Dx1'] * S[1].loc[5,10])+ (procPythonData['Dy1'] * S[1].loc[5,11]) + (procPythonData['Dz1'] * S[1].loc[5,12]) ) * 0.113, #Ino N*m
        'Mz1' : ( (procPythonData['Ax1'] * S[1].loc[6,1]) + (procPythonData['Ay1'] * S[1].loc[6,2]) + (procPythonData['Az1'] * S[1].loc[6,3]) + (procPythonData['Bx1'] * S[1].loc[6,4])+ (procPythonData['By1'] * S[1].loc[6,5])+ (procPythonData['Bz1'] * S[1].loc[6,6]) + (procPythonData['Cx1'] * S[1].loc[6,7]) + (procPythonData['Cy1'] * S[1].loc[6,8]) + (procPythonData['Cz1'] * S[1].loc[6,9]) + (procPythonData['Dx1'] * S[1].loc[6,10])+ (procPythonData['Dy1'] * S[1].loc[6,11]) + (procPythonData['Dz1'] * S[1].loc[6,12]) ) * 0.113, #Ino N*m
        'Sync': procPythonData['Sync'],
        't' : np.arange(0, nReadings) / 200
        }
    return calibPythonData

calibPythonData = readPythonForceData(pythonDataName, calibrationFiles)

plt.plot(calibPythonData['t'] + ProcData['t'][0], calibPythonData['Fz1'], linestyle = '--', linewidth = 4, label = 'Force Plate 0 (Python)')
plt.plot(calibPythonData['t'] + ProcData['t'][0], calibPythonData['Fz0'], linestyle = '--', linewidth = 4, label = 'Force Plate 1 (Python)')

plt.ylabel('Weight (g)')
plt.xlabel('Time (sec)')
plt.legend(loc = 1)
plt.title('Force Plate Recording')
plt.show()

#

intervals = np.diff(ProcData['t'][ProcData['t'] != 0])
sns.distplot(intervals)
plt.xlabel('Reading Interval (s)')
plt.ylabel('Count')
plt.title('NEV Timestamps')
plt.show()

"""
Nonfunctional brpy code...

startTime_s = ProcData['t'][0]
dataLength_s = ProcData['t'][ProcData['t'] != 0][-1] - startTime_s
elecIds = 136
#simi_triggers = getNSxData(dataDir + simiNs5, 98, start_time_s, data_time_s)
filePath = dataDir + simiNs5
def getNSxData(filePath, elecIds, startTime_s, dataLength_s, downsample = 1):
    # Version control
    brpylib_ver_req = "1.3.1"
    if brpylib_ver.split('.') < brpylib_ver_req.split('.'):
        raise Exception("requires brpylib " + brpylib_ver_req + " or higher, please use latest version")

    # Open file and extract headers
    nsx_file = NsxFile(filePath)

    # Extract data - note: data will be returned based on *SORTED* elec_ids, see cont_data['elec_ids']
    channelData = nsx_file.getdata(elecIds, startTime_s, dataLength_s, downsample)
    #nsx_file.extended_headers[-1]
    channelData['data'] = pd.DataFrame(channelData['data'].transpose())
    channelData['t'] = channelData['start_time_s'] + np.arange(channelData['data'].shape[0]) / channelData['samp_per_s']
    channelData['badData'] = dict()
    channelData['spectrum'] = pd.DataFrame({'PSD': [], 't': [], 'fr': [], 'Labels': []})
    channelData['basic_headers'] = nsx_file.basic_header
    channelData['extended_headers'] =  nsx_file.extended_headers
    # Close the nsx file now that all data is out
    nsx_file.close()

    return channelData
"""

#Sync Pulses
timeLimit = 40
procDataMask = np.logical_and(ProcData['t']!=0 , ProcData['t'] < timeLimit)
plt.plot(ProcData['t'][procDataMask], np.array(ProcData['Sync'])[procDataMask], label = 'Sync Pulse (NEV Serial)', marker = 'o')

f = h5py.File(simiTriggersPath,'r')
simiTriggers = np.array(f.get('simiTriggers'))
simiTime = np.arange(len(simiTriggers)) / 3e4
simiMask = simiTime < timeLimit
plt.plot(simiTime[simiMask][::10],simiTriggers[simiMask][::10] / simiTriggers.max(), label = 'Sync Pulse (NS5)')

pythonDataMask = (calibPythonData['t'] + ProcData['t'][0]) < timeLimit
plt.plot(calibPythonData['t'][pythonDataMask] + ProcData['t'][0], np.array(calibPythonData['Sync'])[pythonDataMask], label = 'Sync Pulse (Python)')

plt.legend()

plt.show()
