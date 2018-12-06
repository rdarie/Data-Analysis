import numpy as np
import h5py, pickle

dataDir = 'Y://ENG_Neuromotion_Shared//group//Proprioprosthetics//Data//201811261600-ForceSensorCalibration'
nevDataName = 'datafile002.mat'
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
