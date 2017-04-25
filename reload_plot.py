import matplotlib.pyplot as plt
import pickle

localDir = 'Z:/data/rdarie/tempdata/Data-Analysis/'

with open(localDir + 'spikePlot.pickle', 'rb') as f:
    figs = pickle.load(f)
    ax = figs['spectrum']
ax.show()
x = input()
