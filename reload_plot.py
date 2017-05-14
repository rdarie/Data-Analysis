import matplotlib.pyplot as plt
import pickle, os
import numpy as np

localDir = os.environ['DATA_ANALYSIS_LOCAL_DIR']

with open(localDir + '\spike_LinearSVC_Plot.pickle', 'rb') as f:
    ax = pickle.load(f)

#np.set_printoptions(precision=2)
plt.show()
