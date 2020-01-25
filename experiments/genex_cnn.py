import numpy as np
import pandas as pd
from keras import optimizers
from keras.callbacks import EarlyStopping

from keras.models import Sequential
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Flatten, LSTM

from keras.layers import Dropout
from keras.optimizers import SGD
from keras.regularizers import l2

from sklearn.model_selection import train_test_split
from sklearn import preprocessing, metrics

import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
# smote randomly picks a point from the minority class and compute the k-nearest neighbors for
# this point. The synthetic points are added between the chosen point and its
# neighbors
from sklearn.utils import compute_class_weight, compute_sample_weight
import pydot

smote = SMOTE('minority')

dataset = pd.read_csv('SART2018_HbO.csv')
dataset = dataset.sort_values(by=[' Start time'])

X = dataset.iloc[:, 5:].values
Y = (dataset[' Event Name'])

# deal with nan padding
X = np.nan_to_num(X)
# min-max scale X
mmScaler = preprocessing.MinMaxScaler()
X = mmScaler.fit_transform(X)
# X = X.reshape((-1, 8, 279))  # take every eight row (8 channels) as a sample

# lb = preprocessing.LabelBinarizer()
# Y = lb.fit_transform(np.expand_dims(Y, axis=1))

encoder =preprocessing.OneHotEncoder(categories='auto')
Y = encoder.fit_transform(np.expand_dims(np.asarray(Y), axis=1)).toarray()

# Y = Y[0:-1:8]  # take every eight row as a label


# separate training and test_result set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=3)

#  Using SMOTE oversampling ######################
# X_train = X_train.reshape((len(X_train), -1))  # reshape x so that it can be resampled
# X_train, Y_train = smote.fit_sample(X_train, Y_train)
# X_train = X_train.reshape((len(X_train), 8, -1))  # reshape x back into 8 channel format

# Using Class Weighing ##########################
# classWeight = compute_class_weight('balanced', np.unique(Y), Y)
# classWeight = dict(enumerate(classWeight))

sample_weight = compute_sample_weight('balanced', Y_train)


# build and fit the network
classifier = Sequential()

# classifier.add(Conv1D(filters=32, kernel_size=(3), data_format='channels_first', input_shape=(8, X_train.shape[2]), activation='relu', kernel_regularizer=l2(0.0005)))

X_train, X_test = np.expand_dims(X_train, axis=1), np.expand_dims(X_test, axis=1)
classifier.add(Conv1D(filters=32, kernel_size=(3), data_format='channels_first', input_shape=(1, X_train.shape[2]), activation='relu', kernel_regularizer=l2(0.0005)))

classifier.add(BatchNormalization())
classifier.add(MaxPooling1D(pool_size=2, data_format='channels_first'))

classifier.add(Conv1D(filters=32, kernel_size=(3), data_format='channels_first', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling1D(pool_size=2, data_format='channels_first'))

classifier.add(Conv1D(filters=32, kernel_size=(3), data_format='channels_first', activation='relu'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling1D(pool_size=2, data_format='channels_first'))

classifier.add(Flatten())

classifier.add((Dense(units=64, activation='relu')))
classifier.add(Dropout(rate=0.2))

classifier.add((Dense(units=32, activation='relu')))
classifier.add(Dropout(rate=0.2))

classifier.add((Dense(units=2, activation='softmax')))

epochs = 25000
# adam = optimizers.adam(lr=1e-3, clipnorm=1., decay=1e-2/epochs)  # use half the learning rate as adam optimizer default
sgd = optimizers.SGD(lr=1e-3, decay=1e-2/epochs, momentum=0.9, nesterov=True)
classifier.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1000)
history = classifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=25000, batch_size=16, callbacks=[es], sample_weight=sample_weight)

# plot train history

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test_result'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test_result'], loc='upper left')
plt.show()

y_pred = classifier.predict(X)

# Making the Confusion Matrix
matrix = metrics.confusion_matrix(Y.argmax(axis=1), y_pred.argmax(axis=1))

