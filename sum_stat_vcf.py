#!/usr/bin/env python
# coding: utf-8

# In[16]:





# In[17]:


"""
The updated version: Pi stat

"""

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser(description= 'Generate Summary Statistic from Parsed VCF file')
parser.add_argument('value', type=str, help= 'Parsed VCF file name')
parser.add_argument('subfolder', type=str, help= 'Subfolder name where the parsed VCF file is stored')
args = parser.parse_args()


# In[40]:


val = args.value
sub = args.subfolder


# In[33]:

path_1 = os.getcwd()
path = path_1 + "/" + str(sub) + "/" + str(val)

######
    
    
def pi_stat(matrix, num_strands, window_length, num_stride):
    
    df = pd.DataFrame(matrix)

    strands = num_strands
    stride = num_stride
    data_range = df.shape[1]
    window = window_length
    
    pi_per_SNP = []

    for SNP in range(data_range):
        SNP_list  = list(df.iloc[:, SNP])
        count_0 = SNP_list.count(0)
        count_1 = SNP_list.count(1)

        if count_0 > count_1:
            k = count_0
        else:
            k = count_1
        n = len(SNP_list)
        p_hat = k / n
        pi = (n/(n-1))* 2 * p_hat * (1-p_hat)
        pi_per_SNP.append(pi)

    pi_per_window = []

    for i in range(0, data_range, stride):

        range_1 = i
        range_2 = i + window
        applied_window = window

        if range_2 > data_range:
            range_2 = data_range
            applied_window = range_2 - range_1


        pi_window = sum(pi_per_SNP[range_1:range_2])/applied_window
        pi_per_window.append(pi_window)
    #plt.plot(range(len(pi_per_window)), pi_per_window)
    #plt.title("File: "+str(num_text_file)+" Window Length: "+str(window_length)+" Stride: "+str(num_stride))
    #plt.show()

    return pi_per_window


# In[18]:


"""
The updated version: all stat

"""

import matplotlib.pyplot as plt
from itertools import combinations
import os
import pandas as pd
import numpy as np
import copy
    
    
def all_stat(matrix, num_strands, window_length, num_stride):
    
    
    df = pd.DataFrame(matrix)

    strands = num_strands
    stride = num_stride
    data_range = df.shape[1]
    window = window_length
    h1_per_windows = []
    h12_per_windows = []
    h2h1_per_windows = []
    
    p1_per_window = []
    p2_per_window = []
    p3_per_window = []
    p4_per_window = []
    p5_per_window = []


    for i in range(0, data_range, stride):
        #clear_output(wait=True)
        #print(i)
        all_haplotypes = []
        all_unique_haplotypes = []
        #p1 = 0
        #p2 = 0

        range_1 = i
        range_2 = i + window
        applied_window = window

        if range_2 > data_range:
            range_2 = data_range
            applied_window = range_2 - range_1

        for r in range(strands):
            haplotype = ""
            SNPs = df.iloc[r, range_1: range_2]

            for SNP in SNPs:
                haplotype+= str(SNP)

            all_haplotypes.append(haplotype)

        all_unique_haplotypes = set(all_haplotypes)
        all_unique_haplotypes= list(all_unique_haplotypes)
        freqs = []

        for hap in all_unique_haplotypes:
            freq = all_haplotypes.count(hap)/len(all_haplotypes)
            """
            if p1<x:
                p2 = copy.deepcopy(p1)
                p1 = copy.deepcopy(freq)
            elif c<x<p1:
                p2 = copy.deepcopy(freq)
            """
            freqs.append(freq)
        freqs.sort(reverse = True)
        
        ####
        
        while len(freqs)<5:
                freqs.append(0)



        p1 = freqs[0]
        p2 = freqs[1]
        p3 = freqs[2]
        p4 = freqs[3]
        p5 = freqs[4]
        p1_per_window.append(p1)
        p2_per_window.append(p2)
        p3_per_window.append(p3)
        p4_per_window.append(p4)
        p5_per_window.append(p5)
        
        ###

        sum_term = 0
        for f in range(len(freqs)):
            sum_term += (freqs[f])**2
        h1 = sum_term
        h1_per_windows.append(h1)
        
        ####
        
        h2 = 0
        for f1 in range(1, len(freqs)):
            h2 += (freqs[f1])**2

        h1 = 0
        for f2 in range(len(freqs)):
            h1 += (freqs[f2])**2

        h2h1 = h2/h1
        h2h1_per_windows.append(h2h1)
        
        #####
        
        sum_term = 0
        for f in range(2, len(freqs)):
            sum_term += (freqs[f])**2
        if len(freqs) < 2:
            freqs.append(0)
        h12 = (freqs[0]+freqs[1])**2 + sum_term
        h12_per_windows.append(h12)
        
        

    return h1_per_windows, h12_per_windows, h2h1_per_windows, p1_per_window, p2_per_window, p3_per_window, p4_per_window, p5_per_window


# In[19]:


stats_9 = []


# In[20]:


for i in range(1):
    #if len(pi_list_sweep_train)<1000:
    pi_ = pi_stat(matrix = np.load(path_1 + "/" + str(sub) + "/" + str(val)), num_strands= 198, window_length=10, num_stride=3)
    if pi_ != None:
        stats_9.append(pi_)
        #clear_output(wait=True)
        #print(i)
    else:
        missing.append(i)


# In[21]:


for i in range(1):
    #if len(pi_list_sweep_train)<1000:
    h1_, h12_, h2h1_,p1_, p2_, p3_, p4_, p5_ = all_stat(matrix = np.load(path_1 + "/" + str(sub) + "/" + str(val)), num_strands= 198, window_length=10, num_stride=3)
    if pi_ != None:
        stats_9.append(h1_)
        stats_9.append(h12_)
        stats_9.append(h2h1_)
        stats_9.append(p1_)
        stats_9.append(p2_)
        stats_9.append(p3_)
        stats_9.append(p4_)
        stats_9.append(p5_)
        #clear_output(wait=True)
        #print(i)
    else:
        missing.append(i)


# In[22]:


import numpy as np
chr22 = []
for i in stats_9:
    chr22.append(np.array(i))


# In[23]:


chr22 = np.array(chr22)


# In[24]:


summary_stat_1152 = []


# In[25]:


for i in range(len(stats_9[0])-127):
    row = []
    for j in range(128):
        for k in range(9):
            row.append(stats_9[k][i+j])
            
    summary_stat_1152.append(np.array(row))


# In[26]:


summary_stat_1152 = np.array(summary_stat_1152)


# In[27]:


summary_stat_1152.shape


# In[28]:


np.save('"sumstat_" + str(val)', summary_stat_1152)


# In[ ]:




