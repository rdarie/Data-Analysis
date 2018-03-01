import pickle
from brPY.brpylib import NsxFile, NevFile, brpylib_ver
dataDir = 'W:/ENG_Neuromotion_Shared/group/Starbuck_Bilateral_Recordings'
nevDataName = '/201802191132-Starbuck-Freely_Moving_Force_Plates/201802191132-Starbuck-Freely_Moving_Force_Plates-Array1480_Right_Side-Trial001.nev'

# Open nev data
# Open file and extract headers
nev_file = NevFile(dataDir + nevDataName)
# Extract data and separate out spike data
nevData = nev_file.getdata()
#n= pickle.load(open(, 'rb'))
pickle.dump(nevData, open(dataDir + nevDataName + '.p', 'wb'))

nevDataName = '/201801241426-Freely_Moving_ForcePlates-Array1480_Right_Side-Trail001.nev'

# Open nev data
# Open file and extract headers
nev_file = NevFile(dataDir + nevDataName)
# Extract data and separate out spike data
nevData = nev_file.getdata()
#n= pickle.load(open(, 'rb'))
pickle.dump(nevData, open(dataDir + nevDataName + '.p', 'wb'))
