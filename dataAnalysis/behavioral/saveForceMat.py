import numpy as np
import h5py, pickle

dataDir = 'W:/ENG_Neuromotion_Shared/group/Monkey_Neural_Recordings/Starbuck_Bilateral_Recordings/201802191132-Starbuck-Freely_Moving_Force_Plates'
nevDataName = '201802191132-Starbuck-Freely_Moving_Force_Plates-Array1480_Right_Side-Trial002.mat'
f = h5py.File(dataDir + '/' + nevDataName,'r')

timeStampsHDF = f.get('NEV/Data/SerialDigitalIO/TimeStamp')
timeStamps = np.array(timeStampsHDF)
eventsHDF = f.get('NEV/Data/SerialDigitalIO/UnparsedData')
events = np.array(eventsHDF)
reasonHDF = f.get('NEV/Data/SerialDigitalIO/InsertionReason')
reason = np.array(reasonHDF)

dataDict = {
    'dig_events': {
        'Data' : events,
        'TimeStamps' : timeStamps,
        'Reason' : reason
        }
    }

pickle.dump(dataDict, open(dataDir + '/' + nevDataName + '.p', 'wb'))
