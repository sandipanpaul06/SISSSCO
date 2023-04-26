import numpy as np
import os 



import argparse

parser = argparse.ArgumentParser(description= 'Train the model')

parser.add_argument('train', type=int, help= '# of Training samples')
parser.add_argument('test', type=int, help= '# of Test samples')
parser.add_argument('val', type=int, help= '# of validation samples')

parser.add_argument('W_train', type=str, help= 'Wavelet training file name')
parser.add_argument('M_train', type=str, help= 'Multitaper training file name')
parser.add_argument('S_train', type=str, help= 'S transform training file name')

parser.add_argument('W_test', type=str, help= 'Wavelet testing file name')
parser.add_argument('M_test', type=str, help= 'Multitaper testing file name')
parser.add_argument('S_test', type=str, help= 'S transform testing file name')


parser.add_argument('W_val', type=str, help= 'Wavelet validation file name')
parser.add_argument('M_val', type=str, help= 'Multitaper validation file name')
parser.add_argument('S_val', type=str, help= 'S transform validation file name')


parser.add_argument('y_train', type=str, help= 'Training dataset labels')
parser.add_argument('y_test', type=str, help= 'Testing dataset labels')
parser.add_argument('y_val', type=str, help= 'Validation dataset labels')




args = parser.parse_args()





# In[40]:


s_tr = args.S_train
m_tr = args.M_train
w_tr = args.W_train

s_ts = args.S_test
m_ts = args.M_test
w_ts = args.W_test

s_vl = args.S_val
m_vl = args.M_val
w_vl = args.W_val


tr_n = args.train
ts_n = args.test
vl_n = args.val

y_tr = args.y_train
y_ts = args.y_test
y_vl = args.y_val


# In[33]:

path1 = os.getcwd()








X_train_0 = np.load(path1 + '/TFA/' + s_tr)
X_test_0 = np.load(path1 + '/TFA/' + s_ts)
X_val_0 = np.load(path1 + '/TFA/' + s_vl)
Y_train_0 = np.load(path1 + '/TFA/' + y_tr)
Y_test_0 = np.load(path1 + '/TFA/' + y_ts)
Y_val_0 = np.load(path1 + '/TFA/' + y_vl)


####
tr = np.zeros((2*tr_n, 65, 128, 1))

X_9_stockwell_train = []

for s in range(9):
    for i in range(2*tr_n):
        for j in range(65):
            for k in range(128):
                tr[i][j][k][0] = X_train_0[i][j][k][s]
    X_9_stockwell_train.append(tr)

ts = np.zeros((2*ts_n, 65, 128, 1))

X_9_stockwell_test = []

for s in range(9):
    for i in range(2*ts_n):
        for j in range(65):
            for k in range(128):
                ts[i][j][k][0] = X_test_0[i][j][k][s]
    X_9_stockwell_test.append(ts)
                

vl = np.zeros((2*vl_n, 65, 128, 1))

X_9_stockwell_val = []

for s in range(9):
    for i in range(2*vl_n):
        for j in range(65):
            for k in range(128):
                vl[i][j][k][0] = X_val_0[i][j][k][s]
    X_9_stockwell_val.append(vl)

### free memory space
X_train_0 = None
X_test_0 = None
X_val_0 = None
####

import tensorflow as tf
tf.random.set_seed(221)

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, ReLU, Conv2D, MaxPooling2D, BatchNormalization, Input
from tensorflow.keras.optimizers import Adam

alphas= [x/10 for x in range(0,11,2)]
lambdas = [(10**x) for x in range(-6,5)]


batchSize = 50
numClasses = 2
iterations = 30

##### model0 ####


inputs0_0 = Input(shape=X_9_stockwell_train[0].shape[1:])

m0_0 = inputs0_0
m0_0 = Conv2D(32, (3, 3), padding="same")(m0_0)
m0_0 = BatchNormalization()(m0_0)
m0_0 = Activation('relu') (m0_0)
m0_0 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_0)
m0_0 = Conv2D(32, (3, 3), padding="same")(m0_0)
m0_0 = BatchNormalization()(m0_0)
m0_0 = Activation('relu') (m0_0)
m0_0 = Flatten() (m0_0)
m0_0 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_0)
m0_0 = Activation('relu')(m0_0)
m0_0 = Dropout(0.2) (m0_0)
m0_0 = Dense(numClasses)(m0_0)
m0_0 = Activation('softmax')(m0_0)

model0_0 = Model(inputs0_0, m0_0)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_0.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_0.fit(X_9_stockwell_train[0], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[0], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_0.predict(X_9_stockwell_test[0])


sub_model0_0 = inputs0_0
for layer in model0_0.layers[:-2]: # go through until last layer
    sub_model0_0=layer(sub_model0_0)
    sub_model0_0.trainable = False
    
Model0 = Model(inputs0_0, sub_model0_0)
##### model1 ####


inputs0_1 = Input(shape=X_9_stockwell_train[1].shape[1:])

m0_1 = inputs0_1
m0_1 = Conv2D(32, (3, 3), padding="same")(m0_1)
m0_1 = BatchNormalization()(m0_1)
m0_1 = Activation('relu') (m0_1)
m0_1 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_1)
m0_1 = Conv2D(32, (3, 3), padding="same")(m0_1)
m0_1 = BatchNormalization()(m0_1)
m0_1 = Activation('relu') (m0_1)
m0_1 = Flatten() (m0_1)
m0_1 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_1)
m0_1 = Activation('relu')(m0_1)
m0_1 = Dropout(0.2) (m0_1)
m0_1 = Dense(numClasses)(m0_1)
m0_1 = Activation('softmax')(m0_1)

model0_1 = Model(inputs0_1, m0_1)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_1.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_1.fit(X_9_stockwell_train[1], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[1], Y_val_0), shuffle=False , verbose = 0)



prediction = model0_1.predict(X_9_stockwell_test[1])


sub_model0_1 = inputs0_1
for layer in model0_1.layers[:-2]: # go through until last layer
    sub_model0_1=layer(sub_model0_1)
    sub_model0_1.trainable = False
    
    

##### model2 ####


inputs0_2 = Input(shape=X_9_stockwell_train[2].shape[1:])

m0_2 = inputs0_2
m0_2 = Conv2D(32, (3, 3), padding="same")(m0_2)
m0_2 = BatchNormalization()(m0_2)
m0_2 = Activation('relu') (m0_2)
m0_2 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_2)
m0_2 = Conv2D(32, (3, 3), padding="same")(m0_2)
m0_2 = BatchNormalization()(m0_2)
m0_2 = Activation('relu') (m0_2)
m0_2 = Flatten() (m0_2)
m0_2 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_2)
m0_2 = Activation('relu')(m0_2)
m0_2 = Dropout(0.2) (m0_2)
m0_2 = Dense(numClasses)(m0_2)
m0_2 = Activation('softmax')(m0_2)

model0_2 = Model(inputs0_2, m0_2)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_2.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_2.fit(X_9_stockwell_train[2], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[2], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_2.predict(X_9_stockwell_test[2])


sub_model0_2 = inputs0_2
for layer in model0_2.layers[:-2]: # go through until last layer
    sub_model0_2=layer(sub_model0_2)
    sub_model0_2.trainable = False
    
    
    
##### model3 ####


inputs0_3 = Input(shape=X_9_stockwell_train[3].shape[1:])

m0_3 = inputs0_3
m0_3 = Conv2D(32, (3, 3), padding="same")(m0_3)
m0_3 = BatchNormalization()(m0_3)
m0_3 = Activation('relu') (m0_3)
m0_3 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_3)
m0_3 = Conv2D(32, (3, 3), padding="same")(m0_3)
m0_3 = BatchNormalization()(m0_3)
m0_3 = Activation('relu') (m0_3)
m0_3 = Flatten() (m0_3)
m0_3 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_3)
m0_3 = Activation('relu')(m0_3)
m0_3 = Dropout(0.2) (m0_3)
m0_3 = Dense(numClasses)(m0_3)
m0_3 = Activation('softmax')(m0_3)

model0_3 = Model(inputs0_3, m0_3)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_3.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_3.fit(X_9_stockwell_train[3], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[3], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_3.predict(X_9_stockwell_test[3])


sub_model0_3 = inputs0_3
for layer in model0_3.layers[:-2]: # go through until last layer
    sub_model0_3=layer(sub_model0_3)
    sub_model0_3.trainable = False



##### model4 ####


inputs0_4 = Input(shape=X_9_stockwell_train[4].shape[1:])

m0_4 = inputs0_4
m0_4 = Conv2D(32, (3, 3), padding="same")(m0_4)
m0_4 = BatchNormalization()(m0_4)
m0_4 = Activation('relu') (m0_4)
m0_4 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_4)
m0_4 = Conv2D(32, (3, 3), padding="same")(m0_4)
m0_4 = BatchNormalization()(m0_4)
m0_4 = Activation('relu') (m0_4)
m0_4 = Flatten() (m0_4)
m0_4 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_4)
m0_4 = Activation('relu')(m0_4)
m0_4 = Dropout(0.2) (m0_4)
m0_4 = Dense(numClasses)(m0_4)
m0_4 = Activation('softmax')(m0_4)

model0_4 = Model(inputs0_4, m0_4)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_4.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_4.fit(X_9_stockwell_train[4], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[4], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_4.predict(X_9_stockwell_test[4])


sub_model0_4 = inputs0_4
for layer in model0_4.layers[:-2]: # go through until last layer
    sub_model0_4=layer(sub_model0_4)
    sub_model0_4.trainable = False
    
    
##### model5 ####


inputs0_5 = Input(shape=X_9_stockwell_train[5].shape[1:])

m0_5 = inputs0_5
m0_5 = Conv2D(32, (3, 3), padding="same")(m0_5)
m0_5 = BatchNormalization()(m0_5)
m0_5 = Activation('relu') (m0_5)
m0_5 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_5)
m0_5 = Conv2D(32, (3, 3), padding="same")(m0_5)
m0_5 = BatchNormalization()(m0_5)
m0_5 = Activation('relu') (m0_5)
m0_5 = Flatten() (m0_5)
m0_5 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_5)
m0_5 = Activation('relu')(m0_5)
m0_5 = Dropout(0.2) (m0_5)
m0_5 = Dense(numClasses)(m0_5)
m0_5 = Activation('softmax')(m0_5)

model0_5 = Model(inputs0_5, m0_5)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_5.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_5.fit(X_9_stockwell_train[5], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[5], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_5.predict(X_9_stockwell_test[5])


sub_model0_5 = inputs0_5
for layer in model0_5.layers[:-2]: # go through until last layer
    sub_model0_5=layer(sub_model0_5)
    sub_model0_5.trainable = False
    

##### model6 ####


inputs0_6 = Input(shape=X_9_stockwell_train[6].shape[1:])

m0_6 = inputs0_6
m0_6 = Conv2D(32, (3, 3), padding="same")(m0_6)
m0_6 = BatchNormalization()(m0_6)
m0_6 = Activation('relu') (m0_6)
m0_6 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_6)
m0_6 = Conv2D(32, (3, 3), padding="same")(m0_6)
m0_6 = BatchNormalization()(m0_6)
m0_6 = Activation('relu') (m0_6)
m0_6 = Flatten() (m0_6)
m0_6 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_6)
m0_6 = Activation('relu')(m0_6)
m0_6 = Dropout(0.2) (m0_6)
m0_6 = Dense(numClasses)(m0_6)
m0_6 = Activation('softmax')(m0_6)

model0_6 = Model(inputs0_6, m0_6)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_6.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_6.fit(X_9_stockwell_train[6], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[6], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_6.predict(X_9_stockwell_test[6])


sub_model0_6 = inputs0_6
for layer in model0_6.layers[:-2]: # go through until last layer
    sub_model0_6=layer(sub_model0_6)
    sub_model0_6.trainable = False
    
    
    

##### model7 ####


inputs0_7 = Input(shape=X_9_stockwell_train[7].shape[1:])

m0_7 = inputs0_7
m0_7 = Conv2D(32, (3, 3), padding="same")(m0_7)
m0_7 = BatchNormalization()(m0_7)
m0_7 = Activation('relu') (m0_7)
m0_7 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_7)
m0_7 = Conv2D(32, (3, 3), padding="same")(m0_7)
m0_7 = BatchNormalization()(m0_7)
m0_7 = Activation('relu') (m0_7)
m0_7 = Flatten() (m0_7)
m0_7 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_7)
m0_7 = Activation('relu')(m0_7)
m0_7 = Dropout(0.2) (m0_7)
m0_7 = Dense(numClasses)(m0_7)
m0_7 = Activation('softmax')(m0_7)

model0_7 = Model(inputs0_7, m0_7)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_7.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_7.fit(X_9_stockwell_train[7], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[7], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_7.predict(X_9_stockwell_test[7])


sub_model0_7 = inputs0_7
for layer in model0_7.layers[:-2]: # go through until last layer
    sub_model0_7=layer(sub_model0_7)
    sub_model0_7.trainable = False
    
    
##### model8 ####


inputs0_8 = Input(shape=X_9_stockwell_train[8].shape[1:])

m0_8 = inputs0_8
m0_8 = Conv2D(32, (3, 3), padding="same")(m0_8)
m0_8 = BatchNormalization()(m0_8)
m0_8 = Activation('relu') (m0_8)
m0_8 = MaxPooling2D(pool_size=(2,2), strides=2) (m0_8)
m0_8 = Conv2D(32, (3, 3), padding="same")(m0_8)
m0_8 = BatchNormalization()(m0_8)
m0_8 = Activation('relu') (m0_8)
m0_8 = Flatten() (m0_8)
m0_8 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m0_8)
m0_8 = Activation('relu')(m0_8)
m0_8 = Dropout(0.2) (m0_8)
m0_8 = Dense(numClasses)(m0_8)
m0_8 = Activation('softmax')(m0_8)

model0_8 = Model(inputs0_8, m0_8)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model0_8.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model0_8.fit(X_9_stockwell_train[8], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_stockwell_val[8], Y_val_0), shuffle=False, verbose = 0 )



prediction = model0_8.predict(X_9_stockwell_test[8])


sub_model0_8 = inputs0_8
for layer in model0_8.layers[:-2]: # go through until last layer
    sub_model0_8=layer(sub_model0_8)
    sub_model0_8.trainable = False
    
    
########

X_train_1 = np.load(path1 + '/TFA/' + m_tr)
X_test_1 = np.load(path1 + '/TFA/' + m_ts)
X_val_1 = np.load(path1 + '/TFA/' + m_vl)


a = np.zeros((2*tr_n, 65, 128, 1))

X_9_multitaper_train = []

for s in range(9):
    for i in range(2*tr_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_train_1[i][j][k][s]
    X_9_multitaper_train.append(a)
                
                
a = np.zeros((2*ts_n, 65, 128, 1))
X_9_multitaper_test = []

for s in range(9):
    for i in range(2*ts_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_1[i][j][k][s]
    X_9_multitaper_test.append(a)
                
a = np.zeros((2*vl_n, 65, 128, 1))
X_9_multitaper_val = []

for s in range(9):
    for i in range(2*vl_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_val_1[i][j][k][s]
    X_9_multitaper_val.append(a)

    
### free memory space
X_train_1 = None
X_test_1 = None
X_val_1 = None
####
##### model9 ####


inputs1_0 = Input(shape=X_9_multitaper_train[0].shape[1:])

m1_0 = inputs1_0
m1_0 = Conv2D(32, (3, 3), padding="same")(m1_0)
m1_0 = BatchNormalization()(m1_0)
m1_0 = Activation('relu') (m1_0)
m1_0 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_0)
m1_0 = Conv2D(32, (3, 3), padding="same")(m1_0)
m1_0 = BatchNormalization()(m1_0)
m1_0 = Activation('relu') (m1_0)
m1_0 = Flatten() (m1_0)
m1_0 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m1_0)
m1_0 = Activation('relu')(m1_0)
m1_0 = Dropout(0.2) (m1_0)
m1_0 = Dense(numClasses)(m1_0)
m1_0 = Activation('softmax')(m1_0)

model1_0 = Model(inputs1_0, m1_0)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_0.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_0.fit(X_9_multitaper_train[0], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[0], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_0.predict(X_9_multitaper_test[0])


sub_model1_0 = inputs1_0
for layer in model1_0.layers[:-2]: # go through until last layer
    sub_model1_0=layer(sub_model1_0)
    sub_model1_0.trainable = False



##### model10 ####


inputs1_1 = Input(shape=X_9_multitaper_train[1].shape[1:])

m1_1 = inputs1_1
m1_1 = Conv2D(32, (3, 3), padding="same")(m1_1)
m1_1 = BatchNormalization()(m1_1)
m1_1 = Activation('relu') (m1_1)
m1_1 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_1)
m1_1 = Conv2D(32, (3, 3), padding="same")(m1_1)
m1_1 = BatchNormalization()(m1_1)
m1_1 = Activation('relu') (m1_1)
m1_1 = Flatten() (m1_1)
m1_1 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m1_1)
m1_1 = Activation('relu')(m1_1)
m1_1 = Dropout(0.2) (m1_1)
m1_1 = Dense(numClasses)(m1_1)
m1_1 = Activation('softmax')(m1_1)

model1_1 = Model(inputs1_1, m1_1)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_1.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_1.fit(X_9_multitaper_train[1], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[1], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_1.predict(X_9_multitaper_test[1])


sub_model1_1 = inputs1_1
for layer in model1_1.layers[:-2]: # go through until last layer
    sub_model1_1=layer(sub_model1_1)
    sub_model1_1.trainable = False



##### model11 ####


inputs1_2 = Input(shape=X_9_multitaper_train[2].shape[1:])

m1_2 = inputs1_2
m1_2 = Conv2D(32, (3, 3), padding="same")(m1_2)
m1_2 = BatchNormalization()(m1_2)
m1_2 = Activation('relu') (m1_2)
m1_2 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_2)
m1_2 = Conv2D(32, (3, 3), padding="same")(m1_2)
m1_2 = BatchNormalization()(m1_2)
m1_2 = Activation('relu') (m1_2)
m1_2 = Flatten() (m1_2)
m1_2 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[1])*lambdas[3], l2=alphas[1]*lambdas[3])) (m1_2)
m1_2 = Activation('relu')(m1_2)
m1_2 = Dropout(0.2) (m1_2)
m1_2 = Dense(numClasses)(m1_2)
m1_2 = Activation('softmax')(m1_2)

model1_2 = Model(inputs1_2, m1_2)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_2.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_2.fit(X_9_multitaper_train[2], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[2], Y_val_0), shuffle=False, verbose = 0 )



prediction = model1_2.predict(X_9_multitaper_test[2])


sub_model1_2 = inputs1_2
for layer in model1_2.layers[:-2]: # go through until last layer
    sub_model1_2=layer(sub_model1_2)
    sub_model1_2.trainable = False

############



##### model12 ####

inputs1_3 = Input(shape=X_9_multitaper_train[3].shape[1:])

m1_3 = inputs1_3
m1_3 = Conv2D(32, (3, 3), padding="same")(m1_3)
m1_3 = BatchNormalization()(m1_3)
m1_3 = Activation('relu') (m1_3)
m1_3 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_3)
m1_3 = Conv2D(32, (3, 3), padding="same")(m1_3)
m1_3 = BatchNormalization()(m1_3)
m1_3 = Activation('relu') (m1_3)
m1_3 = Flatten() (m1_3)
m1_3 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_3)
m1_3 = Activation('relu') (m1_3)
m1_3 = Dropout(0.3) (m1_3)
m1_3 = Dense(numClasses)(m1_3)
m1_3 = Activation('softmax')(m1_3)

model1_3 = Model(inputs1_3, m1_3)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_3.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_3.fit(X_9_multitaper_train[3], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[3], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_3.predict(X_9_multitaper_test[3])


sub_model1_3 = inputs1_3
for layer in model1_3.layers[:-2]: # go through until last layer
    sub_model1_3=layer(sub_model1_3)
    sub_model1_3.trainable = False
    
    
    
##### model13 ####

inputs1_4 = Input(shape=X_9_multitaper_train[4].shape[1:])

m1_4 = inputs1_4
m1_4 = Conv2D(32, (3, 3), padding="same")(m1_4)
m1_4 = BatchNormalization()(m1_4)
m1_4 = Activation('relu') (m1_4)
m1_4 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_4)
m1_4 = Conv2D(32, (3, 3), padding="same")(m1_4)
m1_4 = BatchNormalization()(m1_4)
m1_4 = Activation('relu') (m1_4)
m1_4 = Flatten() (m1_4)
m1_4 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_4)
m1_4 = Activation('relu') (m1_4)
m1_4 = Dropout(0.3) (m1_4)
m1_4 = Dense(numClasses)(m1_4)
m1_4 = Activation('softmax')(m1_4)

model1_4 = Model(inputs1_4, m1_4)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_4.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_4.fit(X_9_multitaper_train[4], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[4], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_4.predict(X_9_multitaper_test[4])


sub_model1_4 = inputs1_4
for layer in model1_4.layers[:-2]: # go through until last layer
    sub_model1_4=layer(sub_model1_4)
    sub_model1_4.trainable = False
    
    
    
##### model14 ####

inputs1_5 = Input(shape=X_9_multitaper_train[5].shape[1:])

m1_5 = inputs1_5
m1_5 = Conv2D(32, (3, 3), padding="same")(m1_5)
m1_5 = BatchNormalization()(m1_5)
m1_5 = Activation('relu') (m1_5)
m1_5 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_5)
m1_5 = Conv2D(32, (3, 3), padding="same")(m1_5)
m1_5 = BatchNormalization()(m1_5)
m1_5 = Activation('relu') (m1_5)
m1_5 = Flatten() (m1_5)
m1_5 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_5)
m1_5 = Activation('relu') (m1_5)
m1_5 = Dropout(0.3) (m1_5)
m1_5 = Dense(numClasses)(m1_5)
m1_5 = Activation('softmax')(m1_5)

model1_5 = Model(inputs1_5, m1_5)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_5.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_5.fit(X_9_multitaper_train[5], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[5], Y_val_0), shuffle=False, verbose = 0 )



prediction = model1_5.predict(X_9_multitaper_test[5])


sub_model1_5 = inputs1_5
for layer in model1_5.layers[:-2]: # go through until last layer
    sub_model1_5=layer(sub_model1_5)
    sub_model1_5.trainable = False
    
    
##### model15 ####

inputs1_6 = Input(shape=X_9_multitaper_train[6].shape[1:])

m1_6 = inputs1_6
m1_6 = Conv2D(32, (3, 3), padding="same")(m1_6)
m1_6 = BatchNormalization()(m1_6)
m1_6 = Activation('relu') (m1_6)
m1_6 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_6)
m1_6 = Conv2D(32, (3, 3), padding="same")(m1_6)
m1_6 = BatchNormalization()(m1_6)
m1_6 = Activation('relu') (m1_6)
m1_6 = Flatten() (m1_6)
m1_6 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_6)
m1_6 = Activation('relu') (m1_6)
m1_6 = Dropout(0.3) (m1_6)
m1_6 = Dense(numClasses)(m1_6)
m1_6 = Activation('softmax')(m1_6)

model1_6 = Model(inputs1_6, m1_6)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_6.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_6.fit(X_9_multitaper_train[6], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[6], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_6.predict(X_9_multitaper_test[6])


sub_model1_6 = inputs1_6
for layer in model1_6.layers[:-2]: # go through until last layer
    sub_model1_6=layer(sub_model1_6)
    sub_model1_6.trainable = False



##### model16 ####

inputs1_7 = Input(shape=X_9_multitaper_train[7].shape[1:])

m1_7 = inputs1_7
m1_7 = Conv2D(32, (3, 3), padding="same")(m1_7)
m1_7 = BatchNormalization()(m1_7)
m1_7 = Activation('relu') (m1_7)
m1_7 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_7)
m1_7 = Conv2D(32, (3, 3), padding="same")(m1_7)
m1_7 = BatchNormalization()(m1_7)
m1_7 = Activation('relu') (m1_7)
m1_7 = Flatten() (m1_7)
m1_7 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_7)
m1_7 = Activation('relu') (m1_7)
m1_7 = Dropout(0.3) (m1_7)
m1_7 = Dense(numClasses)(m1_7)
m1_7 = Activation('softmax')(m1_7)

model1_7 = Model(inputs1_7, m1_7)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_7.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_7.fit(X_9_multitaper_train[7], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[7], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_7.predict(X_9_multitaper_test[7])


sub_model1_7 = inputs1_7
for layer in model1_7.layers[:-2]: # go through until last layer
    sub_model1_7=layer(sub_model1_7)
    sub_model1_7.trainable = False
    
    
    
##### model17 ####

inputs1_8 = Input(shape=X_9_multitaper_train[8].shape[1:])

m1_8 = inputs1_8
m1_8 = Conv2D(32, (3, 3), padding="same")(m1_8)
m1_8 = BatchNormalization()(m1_8)
m1_8 = Activation('relu') (m1_8)
m1_8 = MaxPooling2D(pool_size=(2,2), strides=2) (m1_8)
m1_8 = Conv2D(32, (3, 3), padding="same")(m1_8)
m1_8 = BatchNormalization()(m1_8)
m1_8 = Activation('relu') (m1_8)
m1_8 = Flatten() (m1_8)
m1_8 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m1_8)
m1_8 = Activation('relu') (m1_8)
m1_8 = Dropout(0.3) (m1_8)
m1_8 = Dense(numClasses)(m1_8)
m1_8 = Activation('softmax')(m1_8)

model1_8 = Model(inputs1_8, m1_8)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model1_8.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model1_8.fit(X_9_multitaper_train[8], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_multitaper_val[8], Y_val_0), shuffle=False , verbose = 0)



prediction = model1_8.predict(X_9_multitaper_test[8])


sub_model1_8 = inputs1_8
for layer in model1_8.layers[:-2]: # go through until last layer
    sub_model1_8=layer(sub_model1_8)
    sub_model1_8.trainable = False


########

X_train_2 = np.load(path1 + '/TFA/' + w_tr)
X_test_2 = np.load(path1 + '/TFA/' + w_ts)
X_val_2 = np.load(path1 + '/TFA/' + w_vl)


a = np.zeros((2*tr_n, 65, 128, 1))

X_9_wavelet_train = []

for s in range(9):
    for i in range(2*tr_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_train_2[i][j][k][s]
    X_9_wavelet_train.append(a)
                
a = np.zeros((2*ts_n, 65, 128, 1))
X_9_wavelet_test = []

for s in range(9):
    for i in range(2*ts_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_test_2[i][j][k][s]
    X_9_wavelet_test.append(a)
                
a = np.zeros((2*vl_n, 65, 128, 1))
X_9_wavelet_val = []

for s in range(9):
    for i in range(2*vl_n):
        for j in range(65):
            for k in range(128):
                a[i][j][k][0] = X_val_2[i][j][k][s]
    X_9_wavelet_val.append(a)

### free memory space
X_train_2 = None
X_test_2 = None
X_val_2 = None
####
##### model18 ####

inputs2_0 = Input(shape=X_9_wavelet_train[0].shape[1:])

m2_0 = inputs2_0
m2_0 = Conv2D(32, (3, 3), padding="same")(m2_0)
m2_0 = BatchNormalization()(m2_0)
m2_0 = Activation('relu') (m2_0)
m2_0 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_0)
m2_0 = Conv2D(32, (3, 3), padding="same")(m2_0)
m2_0 = BatchNormalization()(m2_0)
m2_0 = Activation('relu') (m2_0)
m2_0 = Conv2D(32, (3, 3), padding="same")(m2_0)
m2_0 = BatchNormalization()(m2_0)
m2_0 = Activation('relu') (m2_0)
m2_0 = Flatten() (m2_0)
m2_0 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_0)
m2_0 = Activation('relu') (m2_0)
m2_0 = Dropout(0.3) (m2_0)
m2_0 = Dense(numClasses)(m2_0)
m2_0 = Activation('softmax')(m2_0)

model2_0 = Model(inputs2_0, m2_0)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_0.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_0.fit(X_9_wavelet_train[0], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[0], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_0.predict(X_9_wavelet_test[0])


sub_model2_0 = inputs2_0
for layer in model2_0.layers[:-2]: # go through until last layer
    sub_model2_0=layer(sub_model2_0)
    sub_model2_0.trainable = False
    


##### model19 ####

inputs2_1 = Input(shape=X_9_wavelet_train[1].shape[1:])

m2_1 = inputs2_1
m2_1 = Conv2D(32, (3, 3), padding="same")(m2_1)
m2_1 = BatchNormalization()(m2_1)
m2_1 = Activation('relu') (m2_1)
m2_1 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_1)
m2_1 = Conv2D(32, (3, 3), padding="same")(m2_1)
m2_1 = BatchNormalization()(m2_1)
m2_1 = Activation('relu') (m2_1)
m2_1 = Conv2D(32, (3, 3), padding="same")(m2_1)
m2_1 = BatchNormalization()(m2_1)
m2_1 = Activation('relu') (m2_1)
m2_1 = Flatten() (m2_1)
m2_1 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_1)
m2_1 = Activation('relu') (m2_1)
m2_1 = Dropout(0.3) (m2_1)
m2_1 = Dense(numClasses)(m2_1)
m2_1 = Activation('softmax')(m2_1)

model2_1 = Model(inputs2_1, m2_1)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_1.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_1.fit(X_9_wavelet_train[1], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[1], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_1.predict(X_9_wavelet_test[1])


sub_model2_1 = inputs2_1
for layer in model2_1.layers[:-2]: # go through until last layer
    sub_model2_1=layer(sub_model2_1)
    sub_model2_1.trainable = False
 


##### model20 ####

inputs2_2 = Input(shape=X_9_wavelet_train[2].shape[1:])

m2_2 = inputs2_2
m2_2 = Conv2D(32, (3, 3), padding="same")(m2_2)
m2_2 = BatchNormalization()(m2_2)
m2_2 = Activation('relu') (m2_2)
m2_2 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_2)
m2_2 = Conv2D(32, (3, 3), padding="same")(m2_2)
m2_2 = BatchNormalization()(m2_2)
m2_2 = Activation('relu') (m2_2)
m2_2 = Conv2D(32, (3, 3), padding="same")(m2_2)
m2_2 = BatchNormalization()(m2_2)
m2_2 = Activation('relu') (m2_2)
m2_2 = Flatten() (m2_2)
m2_2 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_2)
m2_2 = Activation('relu') (m2_2)
m2_2 = Dropout(0.3) (m2_2)
m2_2 = Dense(numClasses)(m2_2)
m2_2 = Activation('softmax')(m2_2)

model2_2 = Model(inputs2_2, m2_2)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_2.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_2.fit(X_9_wavelet_train[2], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[2], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_2.predict(X_9_wavelet_test[2])


sub_model2_2 = inputs2_2
for layer in model2_2.layers[:-2]: # go through until last layer
    sub_model2_2=layer(sub_model2_2)
    sub_model2_2.trainable = False
 
 
##### model21 ####

inputs2_3 = Input(shape=X_9_wavelet_train[3].shape[1:])

m2_3 = inputs2_3
m2_3 = Conv2D(32, (3, 3), padding="same")(m2_3)
m2_3 = BatchNormalization()(m2_3)
m2_3 = Activation('relu') (m2_3)
m2_3 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_3)
m2_3 = Conv2D(32, (3, 3), padding="same")(m2_3)
m2_3 = BatchNormalization()(m2_3)
m2_3 = Activation('relu') (m2_3)
m2_3 = Conv2D(32, (3, 3), padding="same")(m2_3)
m2_3 = BatchNormalization()(m2_3)
m2_3 = Activation('relu') (m2_3)
m2_3 = Flatten() (m2_3)
m2_3 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_3)
m2_3 = Activation('relu') (m2_3)
m2_3 = Dropout(0.3) (m2_3)
m2_3 = Dense(numClasses)(m2_3)
m2_3 = Activation('softmax')(m2_3)

model2_3 = Model(inputs2_3, m2_3)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_3.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_3.fit(X_9_wavelet_train[3], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[3], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_3.predict(X_9_wavelet_test[3])


sub_model2_3 = inputs2_3
for layer in model2_3.layers[:-2]: # go through until last layer
    sub_model2_3=layer(sub_model2_3)
    sub_model2_3.trainable = False
    
    
##### model22 ####

inputs2_4 = Input(shape=X_9_wavelet_train[4].shape[1:])

m2_4 = inputs2_4
m2_4 = Conv2D(32, (3, 3), padding="same")(m2_4)
m2_4 = BatchNormalization()(m2_4)
m2_4 = Activation('relu') (m2_4)
m2_4 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_4)
m2_4 = Conv2D(32, (3, 3), padding="same")(m2_4)
m2_4 = BatchNormalization()(m2_4)
m2_4 = Activation('relu') (m2_4)
m2_4 = Conv2D(32, (3, 3), padding="same")(m2_4)
m2_4 = BatchNormalization()(m2_4)
m2_4 = Activation('relu') (m2_4)
m2_4 = Flatten() (m2_4)
m2_4 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_4)
m2_4 = Activation('relu') (m2_4)
m2_4 = Dropout(0.3) (m2_4)
m2_4 = Dense(numClasses)(m2_4)
m2_4 = Activation('softmax')(m2_4)

model2_4 = Model(inputs2_4, m2_4)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_4.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_4.fit(X_9_wavelet_train[4], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[4], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_4.predict(X_9_wavelet_test[4])


sub_model2_4 = inputs2_4
for layer in model2_4.layers[:-2]: # go through until last layer
    sub_model2_4=layer(sub_model2_4)
    sub_model2_4.trainable = False



##### model23 ####

inputs2_5 = Input(shape=X_9_wavelet_train[5].shape[1:])

m2_5 = inputs2_5
m2_5 = Conv2D(32, (3, 3), padding="same")(m2_5)
m2_5 = BatchNormalization()(m2_5)
m2_5 = Activation('relu') (m2_5)
m2_5 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_5)
m2_5 = Conv2D(32, (3, 3), padding="same")(m2_5)
m2_5 = BatchNormalization()(m2_5)
m2_5 = Activation('relu') (m2_5)
m2_5 = Conv2D(32, (3, 3), padding="same")(m2_5)
m2_5 = BatchNormalization()(m2_5)
m2_5 = Activation('relu') (m2_5)
m2_5 = Flatten() (m2_5)
m2_5 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_5)
m2_5 = Activation('relu') (m2_5)
m2_5 = Dropout(0.3) (m2_5)
m2_5 = Dense(numClasses)(m2_5)
m2_5 = Activation('softmax')(m2_5)

model2_5 = Model(inputs2_5, m2_5)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_5.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_5.fit(X_9_wavelet_train[5], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[5], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_5.predict(X_9_wavelet_test[5])


sub_model2_5 = inputs2_5
for layer in model2_5.layers[:-2]: # go through until last layer
    sub_model2_5=layer(sub_model2_5)
    sub_model2_5.trainable = False
    
    
    
##### model24 ####

inputs2_6 = Input(shape=X_9_wavelet_train[6].shape[1:])

m2_6 = inputs2_6
m2_6 = Conv2D(32, (3, 3), padding="same")(m2_6)
m2_6 = BatchNormalization()(m2_6)
m2_6 = Activation('relu') (m2_6)
m2_6 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_6)
m2_6 = Conv2D(32, (3, 3), padding="same")(m2_6)
m2_6 = BatchNormalization()(m2_6)
m2_6 = Activation('relu') (m2_6)
m2_6 = Conv2D(32, (3, 3), padding="same")(m2_6)
m2_6 = BatchNormalization()(m2_6)
m2_6 = Activation('relu') (m2_6)
m2_6 = Flatten() (m2_6)
m2_6 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_6)
m2_6 = Activation('relu') (m2_6)
m2_6 = Dropout(0.3) (m2_6)
m2_6 = Dense(numClasses)(m2_6)
m2_6 = Activation('softmax')(m2_6)

model2_6 = Model(inputs2_6, m2_6)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_6.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_6.fit(X_9_wavelet_train[6], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[6], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_6.predict(X_9_wavelet_test[6])


sub_model2_6 = inputs2_6
for layer in model2_6.layers[:-2]: # go through until last layer
    sub_model2_6=layer(sub_model2_6)
    sub_model2_6.trainable = False
    


##### model25 ####

inputs2_7 = Input(shape=X_9_wavelet_train[7].shape[1:])

m2_7 = inputs2_7
m2_7 = Conv2D(32, (3, 3), padding="same")(m2_7)
m2_7 = BatchNormalization()(m2_7)
m2_7 = Activation('relu') (m2_7)
m2_7 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_7)
m2_7 = Conv2D(32, (3, 3), padding="same")(m2_7)
m2_7 = BatchNormalization()(m2_7)
m2_7 = Activation('relu') (m2_7)
m2_7 = Conv2D(32, (3, 3), padding="same")(m2_7)
m2_7 = BatchNormalization()(m2_7)
m2_7 = Activation('relu') (m2_7)
m2_7 = Flatten() (m2_7)
m2_7 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_7)
m2_7 = Activation('relu') (m2_7)
m2_7 = Dropout(0.3) (m2_7)
m2_7 = Dense(numClasses)(m2_7)
m2_7 = Activation('softmax')(m2_7)

model2_7 = Model(inputs2_7, m2_7)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_7.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_7.fit(X_9_wavelet_train[7], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[7], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_7.predict(X_9_wavelet_test[7])


sub_model2_7 = inputs2_7
for layer in model2_7.layers[:-2]: # go through until last layer
    sub_model2_7=layer(sub_model2_7)
    sub_model2_7.trainable = False
    


##### model26 ####

inputs2_8 = Input(shape=X_9_wavelet_train[8].shape[1:])

m2_8 = inputs2_8
m2_8 = Conv2D(32, (3, 3), padding="same")(m2_8)
m2_8 = BatchNormalization()(m2_8)
m2_8 = Activation('relu') (m2_8)
m2_8 = MaxPooling2D(pool_size=(2,2), strides=2) (m2_8)
m2_8 = Conv2D(32, (3, 3), padding="same")(m2_8)
m2_8 = BatchNormalization()(m2_8)
m2_8 = Activation('relu') (m2_8)
m2_8 = Conv2D(32, (3, 3), padding="same")(m2_8)
m2_8 = BatchNormalization()(m2_8)
m2_8 = Activation('relu') (m2_8)
m2_8 = Flatten() (m2_8)
m2_8 = Dense(128, kernel_regularizer=tf.keras.regularizers.l1_l2(l1=(1-alphas[4])*lambdas[3], l2=alphas[4]*lambdas[3])) (m2_8)
m2_8 = Activation('relu') (m2_8)
m2_8 = Dropout(0.3) (m2_8)
m2_8 = Dense(numClasses)(m2_8)
m2_8 = Activation('softmax')(m2_8)

model2_8 = Model(inputs2_8, m2_8)

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = model2_8.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = model2_8.fit(X_9_wavelet_train[8], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= (X_9_wavelet_val[8], Y_val_0), shuffle=False , verbose = 0)



prediction = model2_8.predict(X_9_wavelet_test[8])


sub_model2_8 = inputs2_8
for layer in model2_8.layers[:-2]: # go through until last layer
    sub_model2_8=layer(sub_model2_8)
    sub_model2_8.trainable = False
    


####


from tensorflow.keras.layers import *
Model0 = Model(inputs0_0, sub_model0_0)
Model1 = Model(inputs0_1, sub_model0_1)
Model2 = Model(inputs0_2, sub_model0_2)
Model3 = Model(inputs0_3, sub_model0_3)
Model4 = Model(inputs0_4, sub_model0_4)
Model5 = Model(inputs0_5, sub_model0_5)
Model6 = Model(inputs0_6, sub_model0_6)
Model7 = Model(inputs0_7, sub_model0_7)
Model8 = Model(inputs0_8, sub_model0_8)
Model9 = Model(inputs1_0, sub_model1_0)
Model10 = Model(inputs1_1, sub_model1_1)
Model11 = Model(inputs1_2, sub_model1_2)
Model12 = Model(inputs1_3, sub_model1_3)
Model13 = Model(inputs1_4, sub_model1_4)
Model14 = Model(inputs1_5, sub_model1_5)
Model15 = Model(inputs1_6, sub_model1_6)
Model16 = Model(inputs1_7, sub_model1_7)
Model17 = Model(inputs1_8, sub_model1_8)
Model18 = Model(inputs2_0, sub_model2_0)
Model19 = Model(inputs2_1, sub_model2_1)
Model20 = Model(inputs2_2, sub_model2_2)
Model21 = Model(inputs2_3, sub_model2_3)
Model22 = Model(inputs2_4, sub_model2_4)
Model23 = Model(inputs2_5, sub_model2_5)
Model24 = Model(inputs2_6, sub_model2_6)
Model25 = Model(inputs2_7, sub_model2_7)
Model26 = Model(inputs2_8, sub_model2_8)


combined_model = Add()([Model0.output, Model1.output, Model2.output, Model3.output ,Model4.output ,Model5.output , Model6.output, Model7.output, Model8.output, Model9.output, Model10.output, Model11.output, Model12.output, Model13.output, Model14.output, Model15.output, Model16.output, Model17.output, Model18.output, Model19.output, Model20.output, Model21.output, Model22.output, Model23.output, Model24.output, Model25.output, Model26.output])

combined_model.trainable = False
combined_model = Dense(numClasses)(combined_model)
combined_model = Activation('softmax')(combined_model)

finalModel = Model([inputs0_0, inputs0_1, inputs0_2, inputs0_3, inputs0_4, inputs0_5, inputs0_6, inputs0_7, inputs0_8, inputs1_0, inputs1_1, inputs1_2, inputs1_3, inputs1_4, inputs1_5, inputs1_6, inputs1_7, inputs1_8, inputs2_0, inputs2_1, inputs2_2, inputs2_3, inputs2_4, inputs2_5, inputs2_6, inputs2_7, inputs2_8], combined_model)
#Model0.input, Model1.input, Model2.input, Model3.input ,Model4.input ,Model5.input , Model6.input, Model7.input, Model8.input, Model9.input, Model10.input, Model1.input, Model12.input, Model13.input, Model14.input, Model15.input, Model16.input, Model17.input, Model18.input, Model19.input, Model20.input, Model21.input, Model22.input, Model23.input, Model24.input, Model25.input, Model26.input

opt1 = Adam(learning_rate=0.0001, epsilon=1e-06)
history = finalModel.compile(loss="categorical_crossentropy",  metrics=['accuracy'])
history1 = finalModel.fit([X_9_stockwell_train[0], X_9_stockwell_train[1], X_9_stockwell_train[2], X_9_stockwell_train[3], X_9_stockwell_train[4], X_9_stockwell_train[5], X_9_stockwell_train[6], X_9_stockwell_train[7], X_9_stockwell_train[8], X_9_multitaper_train[0], X_9_multitaper_train[1], X_9_multitaper_train[2], X_9_multitaper_train[3], X_9_multitaper_train[4], X_9_multitaper_train[5], X_9_multitaper_train[6], X_9_multitaper_train[7], X_9_multitaper_train[8], X_9_wavelet_train[0], X_9_wavelet_train[1], X_9_wavelet_train[2], X_9_wavelet_train[3], X_9_wavelet_train[4], X_9_wavelet_train[5], X_9_wavelet_train[6], X_9_wavelet_train[7], X_9_wavelet_train[8]], Y_train_0, batch_size=batchSize, epochs=iterations, validation_data= ([X_9_stockwell_val[0], X_9_stockwell_val[1], X_9_stockwell_val[2], X_9_stockwell_val[3], X_9_stockwell_val[4], X_9_stockwell_val[5], X_9_stockwell_val[6], X_9_stockwell_val[7], X_9_stockwell_val[8], X_9_multitaper_val[0], X_9_multitaper_val[1], X_9_multitaper_val[2], X_9_multitaper_val[3], X_9_multitaper_val[4], X_9_multitaper_val[5], X_9_multitaper_val[6], X_9_multitaper_val[7], X_9_multitaper_val[8], X_9_wavelet_val[0], X_9_wavelet_val[1], X_9_wavelet_val[2], X_9_wavelet_val[3], X_9_wavelet_val[4], X_9_wavelet_val[5], X_9_wavelet_val[6], X_9_wavelet_val[7], X_9_wavelet_val[8]], Y_val_0), shuffle=False )

finalModel.save("saved_model_SISSSCO")
prediction = finalModel.predict([X_9_stockwell_test[0], X_9_stockwell_test[1], X_9_stockwell_test[2], X_9_stockwell_test[3], X_9_stockwell_test[4], X_9_stockwell_test[5], X_9_stockwell_test[6], X_9_stockwell_test[7], X_9_stockwell_test[8], X_9_multitaper_test[0], X_9_multitaper_test[1], X_9_multitaper_test[2], X_9_multitaper_test[3], X_9_multitaper_test[4], X_9_multitaper_test[5], X_9_multitaper_test[6], X_9_multitaper_test[7], X_9_multitaper_test[8], X_9_wavelet_test[0], X_9_wavelet_test[1], X_9_wavelet_test[2], X_9_wavelet_test[3], X_9_wavelet_test[4], X_9_wavelet_test[5], X_9_wavelet_test[6], X_9_wavelet_test[7], X_9_wavelet_test[8]])



np.savetxt("test_prediction.csv", prediction, delimiter = ",")

####

