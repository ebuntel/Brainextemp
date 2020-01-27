import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import matplotlib.pyplot as plt


dataset = pd.read_csv('SART2018_HbO.csv')

X = dataset.iloc[:, 5:].values
Y = (dataset[' Event Name'])

# deal with nan padding
X = np.nan_to_num(X)
# min-max scale X
mmScaler = preprocessing.MinMaxScaler()
X = mmScaler.fit_transform(X)
X = np.expand_dims(X, axis=-1)

# binary encode target correct & incorrect
lb = preprocessing.LabelBinarizer()
Y = lb.fit_transform(Y)

# separate training and test_result set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=12)

# build and fit the network
regressiveClassifier = Sequential()

regressiveClassifier.add(LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressiveClassifier.add(Dropout(0.8))

regressiveClassifier.add(LSTM(units=128, return_sequences=True))
regressiveClassifier.add(Dropout(0.5))

regressiveClassifier.add(LSTM(units=128, return_sequences=False))
regressiveClassifier.add(Dropout(0.5))

regressiveClassifier.add((Dense(units=1, activation='sigmoid')))

regressiveClassifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = regressiveClassifier.fit(X_train, Y_train, validation_data=(X_test, Y_test), shuffle=True, epochs=100, batch_size=512)

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