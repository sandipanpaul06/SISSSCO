#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os


# In[33]:


path = "/mnt/beegfs/home/sarnab2020/Datasets/Empirical/Parsing/1/CEU1.vcf"
with open(path, 'r') as file:
    text = file.readlines()


# In[34]:


cols = 0
for i in range(len(text)):
    hap = []
    if text[i][0] != '#':
        cols+=1
        
haps = np.full((198, cols+1), 5)


# In[35]:


counter = 0
missing = []
for i in range(len(text)):
    hap = []
    if text[i][0] != '#':
        counter+= 1
        a = text[i]
        for j in range(len(a)):
            if a[j] == '0' or a[j] == '1':
                if a[j+1] == '|':
                    hap.append(int(a[j]))
                    hap.append(int(a[j+2]))
        if len(hap) == 198:
            for k in range(198):
                haps[k][counter] = hap[k]
        else:
            print('no'+ str(i))
            missing.append(counter)


# In[36]:


pos = []
counter = 0
missing = []
for i in range(len(text)):
    hap = []
    if text[i][0] != '#':
        counter+= 1
        a = text[i]
        for j in range(len(a)):
            if a[j] == '0' or a[j] == '1':
                if a[j+1] == '|':
                    hap.append(int(a[j]))
                    hap.append(int(a[j+2]))
        if len(hap) == 198:
            b= ""
            
            first_t = False
            second_t = False
            j=0
            while not (first_t and second_t):
                if first_t and not second_t:
                    if a[j] == '\t':
                        second_t = True
                    else:
                        b+=str(a[j])
                        
                        
                if not first_t:
                    if a[j] == '\t':
                        first_t = True
                
                j+=1
                
            pos.append(int(b))
        else:
            print('no'+ str(i))
            missing.append(i)


# In[37]:


np.save("chr1_bed_pos", np.array(pos))


# In[38]:


dat = pd.DataFrame(haps)
dataframe = dat.copy()
df = dataframe.drop(0, 1)


# In[39]:


#df


# In[40]:


num = df.to_numpy()
np.save('chr1_bed', num)


# In[ ]:




