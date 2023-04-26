import pandas as pd
import numpy as np
import os
from stockwell import st




import argparse

parser = argparse.ArgumentParser(description= 'Generate Time-frequency images for S transform')
parser.add_argument('sweep_filename', type=str, help= 'Sweep summary statistic filename')
parser.add_argument('neutral_filename', type=str, help= 'Neutral summary statistic filename')
parser.add_argument('train', type=int, help= '# of Training samples')
parser.add_argument('test', type=int, help= '# of Test samples')
parser.add_argument('val', type=int, help= '# of validation samples')

args = parser.parse_args()





# In[40]:


sw_file = args.sweep_filename
nt_file = args.neutral_filename

tr_n = args.train
ts_n = args.test
vl_n = args.val




# In[33]:

path1 = os.getcwd()




"""
The three paths here needs to be changed
"""
sweep = pd.read_csv(path1 + '/Summary_statistics/' + sw_file ,header= None)
neutral = pd.read_csv(path1 + '/Summary_statistics/' + nt_file,header= None)
#test = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/signal_decomposition/22/chr22_1152_bed.npy")

#TEST = pd.DataFrame(test)

All_classes_train_xy = pd.concat([sweep.iloc[:tr_n, :], neutral.iloc[:tr_n, :]])
All_classes_val_xy = pd.concat([sweep.iloc[tr_n:tr_n+vl_n, :], neutral.iloc[tr_n:tr_n+vl_n, :]])
All_classes_test_xy= pd.concat([sweep.iloc[tr_n+vl_n:tr_n+vl_n+ts_n, :], neutral.iloc[tr_n+vl_n:tr_n+vl_n+ts_n, :]])

#All_classes_test_xy= pd.concat([TEST.iloc[:1000, :], TEST.iloc[1000:10000, :]])

label_col = ["Sweep"]*tr_n + ["Neutral"]*tr_n
All_classes_train_xy["Label"] = label_col
All_classes_train_xy= All_classes_train_xy.reset_index()
del All_classes_train_xy['index']
All_classes_train_xy = pd.get_dummies(All_classes_train_xy)
#All_classes_train_xy = All_classes_train_xy.sample(frac=1, random_state=0)


label_col = ["Sweep"]*vl_n + ["Neutral"]*vl_n
All_classes_val_xy["Label"] = label_col
All_classes_val_xy= All_classes_val_xy.reset_index()
del All_classes_val_xy['index']
All_classes_val_xy = pd.get_dummies(All_classes_val_xy)
#All_classes_val_xy = All_classes_val_xy.sample(frac=1, random_state=0)


label_col = ["Sweep"]*ts_n + ["Neutral"]*ts_n
All_classes_test_xy["Label"] = label_col
All_classes_test_xy= All_classes_test_xy.reset_index()
del All_classes_test_xy['index']
All_classes_test_xy = pd.get_dummies(All_classes_test_xy)
#All_classes_test_xy = All_classes_test_xy.sample(frac=1, random_state=0)


summary_statistics = ["pi", "h1", "h12", "h2/h1", "f1", "f2", "f3", "f4", "f5"]

All_classes_stats_train = {}
All_classes_stats_val = {}
All_classes_stats_test = {}

for x in range(9):
    All_classes_stats_train["Stat_{0}".format(summary_statistics[x])] = All_classes_train_xy.iloc[:, x:-2:9]
    All_classes_stats_val["Stat_{0}".format(summary_statistics[x])] = All_classes_val_xy.iloc[:, x:-2:9]
    All_classes_stats_test["Stat_{0}".format(summary_statistics[x])] = All_classes_test_xy.iloc[:, x:-2:9]

import numpy as np

spec_tensor_train = np.empty((2*tr_n, 65, 128, 9))
spec_tensor_val = np.empty((2*vl_n, 65, 128, 9))
spec_tensor_test = np.empty((2*ts_n, 65, 128, 9))




for i in range(All_classes_stats_val["Stat_pi"].shape[0]):
  coefs_list = []
  for j in range(9):
    signal = np.array(All_classes_stats_val["Stat_{0}".format(summary_statistics[j])].iloc[i,:])
    
    
    stock = st.st(signal)
    

    coefs = abs(stock)
    
    

    
    coefs_list.append(coefs)
    
  for k in range(65):
    for l in range(128):
      six_list= []
      for coef in coefs_list:
        six_list.append(coef[k][l])
      spec_tensor_val[i][k][l] = np.array(six_list)
      



for i in range(All_classes_stats_train["Stat_pi"].shape[0]):
  coefs_list = []
  for j in range(9):
    signal = np.array(All_classes_stats_train["Stat_{0}".format(summary_statistics[j])].iloc[i,:])
    
    stock = st.st(signal)
    

    coefs = abs(stock)
    
    

    
    coefs_list.append(coefs)
    
  

  for k in range(65):
    for l in range(128):
      six_list= []
      for coef in coefs_list:
        six_list.append(coef[k][l])
      spec_tensor_train[i][k][l] = np.array(six_list)



for i in range(All_classes_stats_test["Stat_pi"].shape[0]):
  coefs_list = []
  for j in range(9):
    signal = np.array(All_classes_stats_test["Stat_{0}".format(summary_statistics[j])].iloc[i,:])
    
    stock = st.st(signal)
    

    coefs = abs(stock)
    
    

    
    coefs_list.append(coefs)
    
  

  for k in range(65):
    for l in range(128):
      six_list= []
      for coef in coefs_list:
        six_list.append(coef[k][l])
      spec_tensor_test[i][k][l] = np.array(six_list)
      

y_train = All_classes_train_xy.iloc[:, -2:]
y_test = All_classes_test_xy.iloc[:, -2:]
y_val = All_classes_val_xy.iloc[:, -2:]

train_mean = np.empty((1, 65, 128, 9))
train_SD = np.empty((1, 65, 128, 9))


mean_scalogram_sweep = np.empty((65, 128))
mean_scalogram_neutral = np.empty((65, 128))

lab = list(y_train["Label_Neutral"])
#for s in range(len(summary_statistics)):
for s in range(9):
    for row in range(65):
        for col in range(128):
            pixel_list_sweep = []
            pixel_list_neutral = []
            for sim in range(2*tr_n):
                if lab[sim] == 1:
                    pixel_list_neutral.append(spec_tensor_train[sim][row][col][s])
                else:
                    pixel_list_sweep.append(spec_tensor_train[sim][row][col][s])
            
            mean_scalogram_sweep[row][col] = np.mean(pixel_list_sweep)
            mean_scalogram_neutral[row][col] = np.mean(pixel_list_neutral)
        

import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=1, ncols=2)

# generate randomly populated arrays


# find minimum of minima & maximum of maxima
minmin = np.min([np.min(mean_scalogram_neutral), np.min(mean_scalogram_sweep)])
maxmax = np.max([np.max(mean_scalogram_neutral), np.max(mean_scalogram_sweep)])

im1 = axes[0].imshow(mean_scalogram_neutral, vmin=minmin, vmax=maxmax,
                     extent=[0, 150, 30, 1], interpolation='bilinear',  aspect='auto', cmap='bone')
im2 = axes[1].imshow(mean_scalogram_sweep, vmin=minmin, vmax=maxmax,
                     extent=[0, 150, 30, 1], interpolation='bilinear',  aspect='auto', cmap='bone')


axes[0].set_title('Neutral')
axes[1].set_title('Sweep')
# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

plt.savefig("TFA/unstandardized_stockwell.png")


#for s in range(len(summary_statistics)):
for s in range(9):
    for row in range(65):
        for col in range(128):
            pixel_list = []
            for sim in range(All_classes_stats_train["Stat_pi"].shape[0]):
                pixel_list.append(spec_tensor_train[sim][row][col][s])
            pixel_array = np.array(pixel_list)
            train_mean[0][row][col][s] = np.mean(pixel_array)
            train_SD[0][row][col][s] = np.std(pixel_array)
'''

train_mean = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/train_mean_stockwell.npy")
train_SD = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/train_SD_stockwell.npy")
'''
scaled_spec_tensor_train = np.empty((2*tr_n, 65, 128, 9))

for s in range(9):
    for row in range(65):
        for col in range(128):
            for sim in range(All_classes_stats_train["Stat_pi"].shape[0]):
                scaled_spec_tensor_train[sim][row][col][s] = (spec_tensor_train[sim][row][col][s] - train_mean[0][row][col][s])/train_SD[0][row][col][s]

scaled_spec_tensor_test = np.empty((2*ts_n, 65, 128, 9))

for s in range(9):
    for row in range(65):
        for col in range(128):
            for sim in range(All_classes_stats_test["Stat_pi"].shape[0]):
                scaled_spec_tensor_test[sim][row][col][s] = (spec_tensor_test[sim][row][col][s] - train_mean[0][row][col][s])/train_SD[0][row][col][s]

scaled_spec_tensor_val = np.empty((2*vl_n, 65, 128, 9))

for s in range(9):
    for row in range(65):
        for col in range(128):
            for sim in range(All_classes_stats_val["Stat_pi"].shape[0]):
                scaled_spec_tensor_val[sim][row][col][s] = (spec_tensor_val[sim][row][col][s] - train_mean[0][row][col][s])/train_SD[0][row][col][s]
                
np.save("TFA/train_SD_stockwell", train_SD)
np.save("TFA/train_mean_stockwell", train_mean)

standardized_scalogram_sweep = np.empty((65, 128))
standardizzed_scalogram_neutral = np.empty((65, 128))

lab = list(y_train["Label_Neutral"])
#for s in range(len(summary_statistics)):
for s in range(1):
    for row in range(65):
        for col in range(128):
            pixel_list_sweep = []
            pixel_list_neutral = []
            for sim in range(2*tr_n):
                if lab[sim] == 1:
                    pixel_list_neutral.append(scaled_spec_tensor_train[sim][row][col][s])
                else:
                    pixel_list_sweep.append(scaled_spec_tensor_train[sim][row][col][s])
            
            standardized_scalogram_sweep[row][col] = np.mean(pixel_list_sweep)
            standardizzed_scalogram_neutral[row][col] = np.mean(pixel_list_neutral)

fig, axes = plt.subplots(nrows=1, ncols=2)

# generate randomly populated arrays


# find minimum of minima & maximum of maxima
minmin = np.min([np.min(standardizzed_scalogram_neutral), np.min(standardized_scalogram_sweep)])
maxmax = np.max([np.max(standardizzed_scalogram_neutral), np.max(standardized_scalogram_sweep)])

im1 = axes[0].imshow(standardizzed_scalogram_neutral, vmin=minmin, vmax=maxmax,
                     extent=[0, 150, 30, 1], interpolation='bilinear',  aspect='auto', cmap='bone')
im2 = axes[1].imshow(standardized_scalogram_sweep, vmin=minmin, vmax=maxmax,
                     extent=[0, 150, 30, 1], interpolation='bilinear',  aspect='auto', cmap='bone')


axes[0].set_title('Neutral')
axes[1].set_title('Sweep')
# add space for colour bar
fig.subplots_adjust(right=0.85)
cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
fig.colorbar(im2, cax=cbar_ax)

plt.savefig('TFA/Standardized_stockwell.png')

X_train_1 = scaled_spec_tensor_train.copy()
X_test_1 = scaled_spec_tensor_test.copy()
X_val_1 = scaled_spec_tensor_val.copy()
Y_train_1 = np.array(y_train)
Y_test_1 = np.array(y_test)
Y_val_1 = np.array(y_val)

np.save("TFA/X_train_stockwell", X_train_1)
np.save("TFA/X_test_stockwell", X_test_1)
np.save("TFA/X_val_stockwell", X_val_1)
#np.save("Y_train", Y_train_1)
#np.save("Y_test", Y_test_1)
#np.save("Y_val", Y_val_1)
            


