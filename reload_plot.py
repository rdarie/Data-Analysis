import matplotlib.pyplot as plt
import pickle

localDir = 'E:/Google Drive/Github/tempdata/Data-Analysis/'

with open(localDir + 'myplot.pickle', 'rb') as f:
    ax = pickle.load(f)
plt.show()
