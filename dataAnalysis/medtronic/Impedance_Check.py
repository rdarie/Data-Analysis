
# coding: utf-8

# In[1]:
insfileRoot = 'W:/ENG_Neuromotion_Shared/group/Proprioprosthetics/Spinal Electrode\Microleads impedance testing/INS'
insFilePath = insfileRoot + '/trial_1.txt'
impedances = {i:[] for i in range(32)}

# In[2]:

with open(insFilePath) as f:
    for line in f:
        idx = line.find('Test Result Impedance [')
        if idx != -1:
            elecIdx = line.split('[')[1].split(']')[0]
            impVal = line.split('[')[1].split(':')[1]
            impedances[int(elecIdx)].append(float(impVal))


# In[3]:

insFilePath_2 = 'C:/Users/radud/Desktop/NANOZ/INS/trial_2.txt'


# In[4]:

with open(insFilePath_2) as f:
    for line in f:
        idx = line.find('Test Result Impedance [')
        if idx != -1:
            elecIdx = line.split('[')[1].split(']')[0]
            impVal = line.split('[')[1].split(']')[1]
            impedances[int(elecIdx) + 16].append(float(impVal))


# In[ ]:




# In[5]:

import numpy as np
import matplotlib.pyplot as plt

impedanceBar = []
impedanceStd = []
elecIdx = []


# In[6]:

for key, value in impedances.items():
    elecIdx.append(key)
    impedanceBar.append(np.mean(value))
    impedanceStd.append(np.std(value))


# In[7]:

fig, ax = plt.subplots()
rects1 = ax.bar(elecIdx, impedanceBar, width = 0.3, color='r', yerr=impedanceStd)


# In[8]:

plt.show()


# In[ ]:
