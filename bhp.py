from __future__ import absolute_import, division, print_function
import tensorflow as tf 
from tensorflow import keras
import numpy as np 
import pandas as pd

print(tf.__version__)

boston_housing=keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels)=boston_housing.load_data()
order=np.argsort(np.random.random(train_labels.shape))

train_data=train_data[order]
train_labels=train_labels[order]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std

print("Training set: {}".format(train_data.shape))
print("Testing set: {}".format(test_data.shape))

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
                'TAX', 'PTRATIO', 'B', 'LSTAT']


df = pd.DataFrame(train_data, columns=column_names)
print(df.head())

def build_model():
  model = keras.Sequential([
    keras.layers.Dense(64, activation=tf.nn.relu,
                 input_shape=(train_data.shape[1],)),
    keras.layers.Dense(64, activation=tf.nn.relu),
    keras.layers.Dense(1)
  ])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae'])
  return model

model = build_model()
model.summary()

class PrintDot(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs):
		if epoch%100==0:
			print('')
		print('.', end='')

EPOCHS=500

history=model.fit(train_data, train_labels,
			 epochs=EPOCHS, validation_split=0.2,
			 verbose=0, callbacks=[PrintDot()])

import matplotlib.pyplot as plt


def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [1000$]')
  plt.plot(history.epoch, np.array(history.history['mean_absolute_error']),
           label='Train Loss')
  plt.plot(history.epoch, np.array(history.history['val_mean_absolute_error']),
           label = 'Val loss')
  plt.legend()
  plt.ylim([0, 5])
  plt.show()

plot_history(history)
