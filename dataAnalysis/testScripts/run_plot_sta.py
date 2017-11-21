from dataAnalysis.helperFunctions.plot_sta import *
import numpy as np
# import pdb

whichChan = [0,1]
chanLabels = ['EMG', 'trigger']

# try different filters on the spikes
remove_mean = lambda x: x-np.mean(x, axis = 0)
baseline_subtract = lambda x: x-x[0]

#Boris
#experiments = ['Control', 'Perlapine', 'Severed']
#amplitudes = ['75', '100', '110', '120', '125', '130', '140', '150', '160', '170', '175', '200', '225', '250', '275', '300']

#Caligula
experiments = ['Control', 'Perlapine']
amplitudes = ['100','120','140','160','180','200','220','240','260','280',
'300','320','340','360','380','400','420','500','600','700','800','900','1000']

#pdb.set_trace()
plot_sta(whichChan, chanLabels, spike_dur = 10, filter_function = remove_mean,
    debugging = True, updateLog = False, override_ylim = [-0.2,0.2],
    normalize = NORM.NONE)
"""
compare_sta(whichChan, chanLabels, spike_dur = 15, conditions = experiments,
filter_function = remove_mean, debugging = True, override_ylim = 0,
normalize = NORM.NONE)
"""
