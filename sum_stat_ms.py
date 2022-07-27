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

parser = argparse.ArgumentParser(description= 'Generate Summary Statistic from .ms files')
parser.add_argument('pref', type=str, help= '.ms file prefix')
parser.add_argument('class_type', type=int, help= 'Class name. 1 = sweep, 0 = neutral')
parser.add_argument('number', type=int, help= 'Number of .ms files of the chosen class')

args = parser.parse_args()


# In[40]:


val = args.pref
num_ = args.number
class_t = args.class_type
sub = "Sweep" if class_t == 1 else "Neutral"
class_name = sub



# In[33]:

path0 = os.getcwd()
path1 = path0 + "/Datasets"


######


def pi_stat(num_text_file, num_strands, SNP_range, window_length, num_stride):
    
        
    path = path1 + "/"
    path+= sub + "/" + val + "_"
    path+= str(num_text_file)+".ms"
    with open(path, 'r') as file:
        text = file.readlines()
    if text[2][:20] == 'trajectory too bigly':
        return None
    segsites = text[5][11:-2]

    strands = num_strands
    stride = num_stride

    sites = [float(segsites[x*9 : x*9+8])for x in range((len(segsites)+1)//9)]
    data = np.zeros((strands, len(sites)))
    for i in range(strands):
        binary = text[i+6][:-2]
        for j in range(len(binary)):
            data[i][j]= int(binary[j])
    dataset = pd.DataFrame(data)


    dataset.columns = sites
    
    err = float('inf')
    idx= None
    for pos in sites:
        error = abs(pos-0.5)
        if error<err:
            err = error
            idx = sites.index(pos)

    data_range = SNP_range
    window = window_length
    
    if idx > data_range//2 and len(sites) -idx >  data_range//2 :
    
    
    
        df = dataset.iloc[:, idx-data_range//2:idx+data_range//2]
        
        #combs = list(combinations(range(100), 2))
        
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
        #print("pi length: ", len(pi_per_window))
    else:
        return None
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



def all_stat(num_text_file, num_strands, SNP_range, window_length, num_stride):
    
        
    path = path1 + "/"
    path+= sub +  "/" + val + "_"
    path+= str(num_text_file)+".ms"
    with open(path, 'r') as file:
        text = file.readlines()
    if text[2][:20] == 'trajectory too bigly':
        return None
    
    segsites = text[5][11:-2]

    strands = num_strands
    stride = num_stride

    sites = [float(segsites[x*9 : x*9+8])for x in range((len(segsites)+1)//9)]
    data = np.zeros((strands, len(sites)))
    for i in range(strands):
        binary = text[i+6][:-2]
        for j in range(len(binary)):
            data[i][j]= int(binary[j])
    dataset = pd.DataFrame(data)


    dataset.columns = sites
    
    err = float('inf')
    idx= None
    for pos in sites:
        error = abs(pos-0.5)
        if error<err:
            err = error
            idx = sites.index(pos)

    data_range = SNP_range
    window = window_length
    h1_per_window = []
    h12_per_window = []
    h2h1_per_window = []
    p1_per_window = []
    p1_per_window = []
    p2_per_window = []
    p3_per_window = []
    p4_per_window = []
    p5_per_window = []
    
    if idx > data_range//2 and len(sites) -idx >  data_range//2 :
    
    
    
        df = dataset.iloc[:, idx-data_range//2:idx+data_range//2]
        
        
        h1_per_windows = []
        
        
        for i in range(0, data_range, stride):
            
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
            h1_per_window.append(h1)
            

            ####

            h2 = 0
            for f1 in range(1, len(freqs)):
                h2 += (freqs[f1])**2

            h1 = 0
            for f2 in range(len(freqs)):
                h1 += (freqs[f2])**2

            h2h1 = h2/h1
            h2h1_per_window.append(h2h1)

            #####

            sum_term = 0
            for f in range(2, len(freqs)):
                sum_term += (freqs[f])**2
            if len(freqs) < 2:
                freqs.append(0)
            h12 = (freqs[0]+freqs[1])**2 + sum_term
            h12_per_window.append(h12)

    #print("p1 length: ", len(p1_per_window))
    #print("h1 length: ", len(h1_per_window))
    else:
        return None

    return h1_per_window, h12_per_window, h2h1_per_window, p1_per_window, p2_per_window, p3_per_window, p4_per_window, p5_per_window

# In[19]:

pi_all = []


# In[20]:


for i in range(num_):
    #if len(pi_list_sweep_train)<1000:
    pi_ = pi_stat(num_text_file = i+1 ,num_strands= 198, SNP_range = 400, window_length=10, num_stride=3)
    if pi_ != None :
        pi_all.append(pi_[3:-3])
        


# In[21]:


h1_all = [] 
h12_all = [] 
h2h1_all = []
p1_all = [] 
p2_all = [] 
p3_all = [] 
p4_all = [] 
p5_all = []

for i in range(num_):
    #if len(pi_list_sweep_train)<1000:
    h1_, h12_, h2h1_,p1_, p2_, p3_, p4_, p5_ = all_stat(num_text_file = i+1 ,num_strands= 198, SNP_range = 400, window_length=10, num_stride=3)
    if h1_ != None:
        h1_all.append(h1_[3:-3])
        h12_all.append(h12_[3:-3])
        h2h1_all.append(h2h1_[3:-3])
        p1_all.append(p1_[3:-3])
        p2_all.append(p2_[3:-3])
        p3_all.append(p3_[3:-3])
        p4_all.append(p4_[3:-3])
        p5_all.append(p5_[3:-3])
        






all_summary_stats = []
all_summary_stats.append(pd.DataFrame(pi_all))
all_summary_stats.append(pd.DataFrame(h1_all))
all_summary_stats.append(pd.DataFrame(h12_all))
all_summary_stats.append(pd.DataFrame(h2h1_all))
all_summary_stats.append(pd.DataFrame(p1_all))
all_summary_stats.append(pd.DataFrame(p2_all))
all_summary_stats.append(pd.DataFrame(p3_all))
all_summary_stats.append(pd.DataFrame(p4_all))
all_summary_stats.append(pd.DataFrame(p5_all))


nump = np.zeros((len(pi_all) , 1152))
df = pd.DataFrame(nump)

stat = 0
for i in all_summary_stats:
    for j in range(128):
        pos = stat + len(all_summary_stats) * j
        df[pos] = list(i.iloc[:, j])
    stat+=1
print("dataset shape : " , df.shape)
df.to_csv(path0 + '/Summary_statistics/' +'training_'+ class_name+ '.csv', index=False, header= False)
