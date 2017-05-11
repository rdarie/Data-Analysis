import matplotlib.pyplot as plt
import pickle, os
import numpy as np

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

with open(localDir + '\spectrumValidationCurvedownSampler_linDis_winLen_0.1_stepLen_0.1_from_scipy.pickle', 'rb') as f:
    ax = pickle.load(f)

#np.set_printoptions(precision=2)
plt.show()
