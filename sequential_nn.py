# libraries used for preparing and processing the dataset
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# libraries used for sequential model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy

# these will hold corresponding samples and labels for the dataset
# samples = x values 
# labels = y values
# dataset = (x,y) pair. For eg. In a 0-9 numbers dataset x will be the picture of a digit, and y will be the label telling us what number is actually in the picture
train_labels = []
train_samples = []

# creating the data ourselves for simplicity 
''' Dataset explainantion
    An experimental drug was tested on individuals from ages 13 to 100 in a clinical trial
    The trial had 2100 participants. half were under 65 years old, half were 65 years or older. 
    Around 95% of patients 65 or older experienced side effects. 
    Aroung 95% of patients under 65 experienced no side effects. 
'''

for i in range(50):
    # generating 5% experiencing side effects from <65
    random_younger = randint(13,64)
    train_samples.append(random_younger)
    train_labels.append(1) # 1 here means side effects were experienced

    # generating 5% not experiencing side effects from >=65
    random_older= randint(65,100)
    train_samples.append(random_older)
    train_labels.append(0) # 0 here means side effects were not experienced

for i in range(1000):
    # generating 95% not experiencing side effects from <65
    train_samples.append(randint(3,64))
    train_labels.append(0)

    # generating 95% experiencing side effects from >=65
    train_samples.append(randint(65,100))
    train_labels.append(1)

# processing the data to pass it to the fit function. fit function is a keras function. 
# one datatype it accepts is numpy so we convert our data to a numpy array
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
train_labels, train_samples = shuffle(train_labels, train_samples)

# it is also a good idea to normalize the data for faster convergence
scaler = MinMaxScaler(feature_range=(0,1)) # rescaling from 0-1 from 13-100
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1,1)) # reshaping for fit_transform function

'''
# uncomment the following lines if you want to run this code on GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print("Num GPUs Available: ", len(physical_devices))
tf.config.experimental.set_memory_growth(physical_devices[0],True)
'''

model = Sequential([
    Dense(units=16, input_shape=(1,),activation='relu'), # (1,) is the shape of the manual input layer
    Dense(units=32,activation='relu'), # hidden layer with 32 cells
    Dense(units=2,activation='softmax') # output layer giving us probabilities for each output class
])

# Dense = Fully connected layer 
model.summary() # shows summary of the created NN

# preparing the model for training
model.compile(optimizer= Adam(learning_rate=0.0001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# training the model using fit() function
model.fit(x=scaled_train_samples,y=train_labels,batch_size=10,epochs=30,shuffle=True,verbose=2)

