import seaborn as sns
import numpy as np
import pandas as pd
import argparse
from helper_functions import *
import re

defaultArgs = ['bestSpikeLDA.pickle',
    'bestSpikeSVMLZ.pickle',
    'bestSpikeSVMRBFZ.pickle',
    'bestSpectrumLDA_DownSampled.pickle',
    'bestSpectrumSVMLZ_DownSampled.pickle']
"""
defaultArgs = ['featureSelected_bestSpikeLDA.pickle',
    'featureSelected_bestSpikeSVMLZ.pickle',
    'featureSelected_bestSpectrumLDA_DownSampled.pickle',
    'featureSelected_bestSpectrumSVMLZ_DownSampled.pickle']
"""
parser = argparse.ArgumentParser()
parser.add_argument('--model', nargs='*', default = defaultArgs)
args = parser.parse_args()
argModel = args.model

results = {'model': [], 'score':[]}
for name in argModel:
    modelFileName = '/' + name
    estimator, estimatorInfo, whichChans, maxFreq = getEstimator(modelFileName)
    idx = np.where(estimatorInfo['rank_test_score'] == 1)[0][0]
    pattern = re.compile('split\d_test_score')
    for key in estimatorInfo.keys():
        if pattern.match(key):
            print(key)
            results['model'].append(name.split('.')[0])
            results['score'].append(estimatorInfo[key][idx])

resultsDf = pd.DataFrame(results)
ax = sns.barplot(x="model", y="score", data=resultsDf)
plt.show()
