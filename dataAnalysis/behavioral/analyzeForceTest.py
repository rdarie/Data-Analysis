import pickle, sys, math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from brpylib import NsxFile, NevFile, brpylib_ver
dataDir = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Data/20180122-ForcePlateTest'
pythonDataName = '/20180122_ForceTest002.p'
nevDataName = '/20180122_ForceTest002.nev.p'

sns.set(font_scale=1.5)
# Open nev data
nevData= pickle.load(open(dataDir + nevDataName, 'rb'))
nPoints = 3
addressIdx = range(1,48*nPoints,2)
matchStr = ['f123456789abba9876543210', 'f123456789abba987654321f', '0123456789abba987654321f', '0123456789abba9876543210']
idNibbles = '' # will hold bits 4-7 assuming the reading frame starts with 2

for idx in addressIdx:
    highByte = format(nevData['dig_events']['Data'][0][idx], '02x')
    #print(highByte)
    idNibbles = idNibbles + highByte[0]

addressStartID = None
if any([combo in idNibbles for combo in matchStr]):
    for combo in matchStr:
        addressStartID = idNibbles.find(combo)
        if addressStartID != -1:
            break
else:
    addressIdx = range(2,48*nPoints,2)
    idNibbles = '' # will hold bits 4-7 assuming the reading frame starts with 2

    for idx in addressIdx:
        highByte = format(nevData['dig_events']['Data'][0][idx], '02x')
        #print(highByte)
        idNibbles = idNibbles + highByte[0]

    addressStartID = None
    if any([combo in idNibbles for combo in matchStr]):
        for combo in matchStr:
            addressStartID = idNibbles.find(combo)
            if addressStartID != -1:
                break

assert addressStartID != -1

startIdx = addressIdx[addressStartID] - 1
format(nevData['dig_events']['Data'][0][startIdx], '02x')

addressIdxRange = range(1, 48, 2)
channelNames = ['Ax0', 'Ay0', 'Az0', 'Bx0', 'By0', 'Bz0', 'Cx0', 'Cy0', 'Cz0', 'Dx0', 'Dy0', 'Dz0',
    'Ax1', 'Ay1', 'Az1', 'Bx1', 'By1', 'Bz1', 'Cx1', 'Cy1', 'Cz1', 'Dx1', 'Dy1', 'Dz1']
channelLookup = {i : channelNames[count] for count, i in enumerate(addressIdxRange)}

nReadings = math.floor(len(nevData['dig_events']['Data'][0]) / 48)
#nReadings = 12500
t = np.arange(0, nReadings) * 1 / 200
blockIdxRange = range(0,48*nReadings,48) #48 blocks per reading
NevProcData = {name : [0 for i in range(nReadings)] for name in channelNames}
# Collect nReadings worth of data:

for readingIdx, blockIdx in enumerate(blockIdxRange):
    for addressIdx in addressIdxRange:
        lowByte = nevData['dig_events']['Data'][0][blockIdx + addressIdx - 1 + startIdx]
        highByte = nevData['dig_events']['Data'][0][blockIdx + addressIdx + startIdx]
        lowByteStr = format(lowByte, '02x')
        highByteStr = format(highByte, '02x')
        #print(lowByteStr + highByteStr)
        NevProcData[channelLookup[addressIdx]][readingIdx] = int(highByteStr[1] + lowByteStr[0] + lowByteStr[1], 16)


"""
Zero the measurements
"""
for key in NevProcData:
    NevProcData[key] = np.asarray(NevProcData[key]) - int(round(np.mean(NevProcData[key][:1000])))

#Load calibration data
S = [0,1]
S[1] = pd.read_csv('C:/Users/Radu/Documents/GitHub/Data-Analysis/D1550_1.acl', names = range(1,13), index_col = False, header = None).reindex(index = range(1,7))
S[0] = pd.read_csv('C:/Users/Radu/Documents/GitHub/Data-Analysis/D1551_2.acl', names = range(1,13), index_col = False).reindex(index = range(1,7))

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
    }

standardDeviation = {}
for key in ProcData:
    standardDeviation[key] = np.std(ProcData[key][:1000])
"""
plt.plot(NevProcData['Az1'] + NevProcData['Bz1'] + NevProcData['Cz1'] + NevProcData['Dz1'])
plt.plot(NevProcData['Az0'] + NevProcData['Bz0'] + NevProcData['Cz0'] + NevProcData['Dz0'])
"""
plt.plot(t, ProcData['Fz0'], label = 'Force Plate 0')
plt.plot(t, ProcData['Fz1'], label = 'Force Plate 1')

plt.ylabel('Weight (g)')
plt.xlabel('Time (sec)')
plt.title('Force Plate Validation')
plt.legend(loc = 1)
plt.show()



# open data saved via python plt.show()
pythonData = pickle.load(open(dataDir + pythonDataName, 'rb'))
procPythonData = {name : [0 for i in range(nReadings)] for name in channelNames}
addressIdxRange = range(1,48,2)

for readingIdx, datum in enumerate(pythonData[:nReadings]):
    for addressIdx in addressIdxRange:
        lowByte = datum[addressIdx - 1]
        highByte = datum[addressIdx]
        lowByteStr = format(lowByte, '02x')
        highByteStr = format(highByte, '02x')
        #print(lowByteStr + highByteStr)
        procPythonData[channelLookup[addressIdx]][readingIdx] = int(highByteStr[1] + lowByteStr[0] + lowByteStr[1], 16)

"""
plt.plot(procPythonData['Az1'])
plt.plot(procPythonData['Az0'])
plt.plot(procPythonData['Az1'] + procPythonData['Bz1'] + procPythonData['Cz1'] + procPythonData['Dz1'])
plt.plot(procPythonData['Az0'] + procPythonData['Bz0'] + procPythonData['Cz0'] + procPythonData['Dz0'])
"""
plt.show()

#plt.plot(np.diff(nevData['dig_events']['TimeStamps'][0]))
#plt.show()
