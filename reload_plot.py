import matplotlib.pyplot as plt
import pickle

localDir = 'Z:/data/rdarie/tempdata/'

with open(localDir + 'mySpectrumPlot.pickle', 'rb') as f:
    ax = pickle.load(f)
plt.show()
