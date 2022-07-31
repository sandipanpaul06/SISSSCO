import numpy as np




mport argparse

parser = argparse.ArgumentParser(description= 'Test model on empirical data')
parser.add_argument('S_test', type=str, help= 'S transform empirical file name')
parser.add_argument('M_test', type=str, help= 'Multitaper empirical file name')
parser.add_argument('W_test', type=str, help= 'Wavelet empirical file name')

parser.add_argument('test', type=int, help= 'Test samples')


args = parser.parse_args()


# In[40]:




s_ts = args.S_test
m_ts = args.M_test
w_ts = args.W_test



ts_n = args.test




# In[33]:

path1 = os.getcwd()







X_test_0 = np.load(path1 + '/TFA/' + s_ts)


####
ts = np.zeros((2*ts_n, 65, 128, 1))

X_9_stockwell_test = []

for s in range(9):
    for i in range(2*ts_n):
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

X_test_1 = np.load(path1 + '/TFA/' + m_ts)
                
a = np.zeros((2*ts_n, 65, 128, 1))
X_9_multitaper_test = []

for s in range(9):
    for i in range(2*ts_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_1[i][j][k][s]
                X_9_multitaper_test.append(a)
                
########

X_test_2 = np.load(path1 + '/TFA/' + w_ts)
a = np.zeros((2*ts_n, 65, 128, 1))
X_9_wavelet_test = []

for s in range(9):
    for i in range(2*ts_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_2[i][j][k][s]
                X_9_wavelet_test.append(a)
                
####
finalModel = tf.keras.models.load_model(path1 + "/saved_model_SISSSCO")

prediction = finalModel.predict([X_9_stockwell_test[0], X_9_stockwell_test[1], X_9_stockwell_test[2], X_9_stockwell_test[3], X_9_stockwell_test[4], X_9_stockwell_test[5], X_9_stockwell_test[6], X_9_stockwell_test[7], X_9_stockwell_test[8], X_9_multitaper_test[0], X_9_multitaper_test[1], X_9_multitaper_test[2], X_9_multitaper_test[3], X_9_multitaper_test[4], X_9_multitaper_test[5], X_9_multitaper_test[6], X_9_multitaper_test[7], X_9_multitaper_test[8], X_9_wavelet_test[0], X_9_wavelet_test[1], X_9_wavelet_test[2], X_9_wavelet_test[3], X_9_wavelet_test[4], X_9_wavelet_test[5], X_9_wavelet_test[6], X_9_wavelet_test[7], X_9_wavelet_test[8]])


np.savetxt("prediction_empirical.csv", prediction, delimiter = ",")

####

