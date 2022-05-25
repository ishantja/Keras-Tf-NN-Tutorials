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
import os.path

# libraries used for visualizing test accuracy 
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt 

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
# uncomment the following line if you want to see the result without validation
# model.fit(x=scaled_train_samples,y=train_labels,batch_size=10,epochs=30,shuffle=True,verbose=2)

'''
validation, in the easiest possible way, can be done by separating a chunk of the original data as a validation set
this separated set can be used to test the accuracy of the model after training
this allows us to see if our model is overfitting 
meaning the model only performs well in the training data
'''

# two ways to create validation for sequential nn
# 1. pass a validation_data argument on fit() function
# 2. let Keras create the data

model.fit(x=scaled_train_samples,y=train_labels,validation_split=0.1,batch_size=10,epochs=30,shuffle=True,verbose=2)
# last 10% of the dataset is separated for validation before training
# shuffle happens after the data is split. So make sure that the data is shuffled beforehand. 
# if the training and validation accuracy, loss are similar, we can say that our model is generalizing well on unseen data

# inferencing from a trained model 
# deploying model on test data to again make sure model can generalize well
# test data has been created in the same way as train data

test_labels = []
test_samples = []

for i in range(10):
    # generating 5% experiencing side effects from <65
    random_younger =     test_samples.append(randint(13,64))
    test_labels.append(1) # 1 here means side effects were experienced

    # generating 5% not experiencing side effects from >=65
    test_samples.append(randint(65,100))
    test_labels.append(0) # 0 here means side effects were not experienced

for i in range(200):
    # generating 95% not experiencing side effects from <65
    test_samples.append(randint(3,64))
    test_labels.append(0)

    # generating 95% experiencing side effects from >=65
    test_samples.append(randint(65,100))
    test_labels.append(1)

test_labels = np.array(test_labels)
test_samples = np.array(test_samples)
test_labels, test_samples = shuffle(test_labels, test_samples) 
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1,1))

# obtaining predictions from trained model 
predictions = model.predict(x=scaled_test_samples, batch_size = 10, verbose =0)
# uncomment to see output probability of each class 
# for i in predictions: 
#     print(i)
rounded_predictions = np.argmax(predictions,axis = -1)
for i in rounded_predictions:
    if i == 0:
        print("Side effects absent")
    else: 
        print("Side effects present")

# visualizing accuracy reading wrt test labels using confusion matrix
cm = confusion_matrix(y_true = test_labels, y_pred = rounded_predictions) 

# function copied from scikit-learn website 
def plot_confusion_matrix(cm, classes,


                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm_plot_labels = ['no_side_effects', 'had_side_effects']
plot_confusion_matrix(cm=cm, classes = cm_plot_labels, title='Confusion Matrix')
plt.show()

# save/load keras sequential model 
if os.path.isfile('model/medical_trial_model.h5') is False: 
    model.save('models/medical_trial_model.h5')
# saves the architecture of the model, weights, training config, optimizer state

# to save only the model architecture, use the to_json 
json_string = model.to_json()
text_file = open("models/model_architecture.txt","w")
text_file.write(json_string)
text_file.close()

# to save as YAML
# yaml_string = model.to_yaml()

# only saving weights
if os.path.isfile('model/medical_trial_model_weights.h5') is False: 
    model.save_weights('models/medical_trial_model_weights.h5')
