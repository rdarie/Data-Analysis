import matplotlib.pyplot as plt
import pickle
import numpy as np

localDir = 'Z:/data/rdarie/tempdata/Data-Analysis/'

with open(localDir + 'spectrumSVC(C=1, caDecisionBoundaryPlot.pickle', 'rb') as f:
    ax = pickle.load(f)

#np.set_printoptions(precision=2)
plt.show()
