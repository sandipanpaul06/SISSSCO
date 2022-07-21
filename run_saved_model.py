import numpy as np
X_test_0 = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/signal_decomposition/22/X_test_stockwell_emp_0.npy")


####
ts = np.zeros((10000, 65, 128, 1))

X_9_stockwell_test = []

for s in range(9):
    for i in range(10000):
        for j in range(65):
            for k in range(128):
                ts[i][j][k][0] = X_test_0[i][j][k][s]
                X_9_stockwell_test.append(ts)
                


import tensorflow as tf
tf.random.set_seed(221)
#import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam



########

X_test_1 = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/signal_decomposition/22/X_test_multitaper_emp_0.npy")
                
a = np.zeros((10000, 65, 128, 1))
X_9_multitaper_test = []

for s in range(9):
    for i in range(10000):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_1[i][j][k][s]
                X_9_multitaper_test.append(a)
                
########

X_test_2 = np.load("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/signal_decomposition/22/X_test_wavelet_emp_0.npy")
a = np.zeros((10000, 65, 128, 1))
X_9_wavelet_test = []

for s in range(9):
    for i in range(10000):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_2[i][j][k][s]
                X_9_wavelet_test.append(a)
                
####
finalModel = tf.keras.models.load_model("/mnt/beegfs/home/sarnab2020/Datasets/Empirical/saved_model/saved_27CMM")

prediction = finalModel.predict([X_9_stockwell_test[0], X_9_stockwell_test[1], X_9_stockwell_test[2], X_9_stockwell_test[3], X_9_stockwell_test[4], X_9_stockwell_test[5], X_9_stockwell_test[6], X_9_stockwell_test[7], X_9_stockwell_test[8], X_9_multitaper_test[0], X_9_multitaper_test[1], X_9_multitaper_test[2], X_9_multitaper_test[3], X_9_multitaper_test[4], X_9_multitaper_test[5], X_9_multitaper_test[6], X_9_multitaper_test[7], X_9_multitaper_test[8], X_9_wavelet_test[0], X_9_wavelet_test[1], X_9_wavelet_test[2], X_9_wavelet_test[3], X_9_wavelet_test[4], X_9_wavelet_test[5], X_9_wavelet_test[6], X_9_wavelet_test[7], X_9_wavelet_test[8]])

np.save("prediction_empirical_27CMM_withSavedModel_0", prediction)

####

