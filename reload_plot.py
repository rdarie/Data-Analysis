import matplotlib.pyplot as plt
import pickle

localDir = 'Z:/data/rdarie/tempdata/Data-Analysis/'

with open(localDir + 'spikeConfusionMatrix.pickle', 'rb') as f:
    ax = pickle.load(f)
plt.show()
