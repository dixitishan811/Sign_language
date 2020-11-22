import pandas as pd
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import csv
from keras.optimizers import SGD
import warnings
warnings.filterwarnings('ignore')
X_train = pd.read_csv("sign_mnist_train.csv") 
X_test = pd.read_csv("sign_mnist_test.csv") 

y_train=X_train['label']
y_test=X_test['label']

    
X_train = np.array(X_train.iloc[:,1:])
X_train = np.array([np.reshape(i, (28,28)) for i in X_train])/255
X_test = np.array(X_test.iloc[:,1:])
X_test = np.array([np.reshape(i, (28,28)) for i in X_test])/255
num_classes = 26
y_train = np.array(y_train).reshape(-1)
y_test = np.array(y_test).reshape(-1)
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
X_train = X_train.reshape((27455, 28, 28, 1))
X_test = X_test.reshape((7172, 28, 28, 1))





classifier = Sequential()
classifier.add(Conv2D(filters=8, kernel_size=(3,3),strides=(1,1),padding='same',input_shape=(28,28,1),activation='relu', data_format='channels_last'))
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Conv2D(filters=16, kernel_size=(3,3),strides=(1,1),padding='same',activation='relu'))
classifier.add(Dropout(0.5))
classifier.add(MaxPooling2D(pool_size=(4,4)))
classifier.add(Dense(128, activation='relu'))
classifier.add(Flatten())
classifier.add(Dense(26, activation='softmax'))

classifier.compile(optimizer='SGD', loss='categorical_crossentropy', metrics=['accuracy'])




classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
classifier.fit(X_train, y_train, epochs=50, batch_size=100)




accuracy = classifier.evaluate(x=X_test,y=y_test,batch_size=32)
print("Accuracy: ",accuracy[1])


classifier.save('CNN.h5')



classifier.summary()





