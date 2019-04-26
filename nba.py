import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils

T = 10   # Number of time steps
N = 5   # Number of players
F = 10  # Number of features
B = 128 # Batch size

home_team = np.random.rand((N, F, T))
away_team = np.random.rand((N, F, T))

# Concatenation to form input
data = np.hconcat((home_team, away_team))

# Create neural network
model = Sequential()


num_filters = 512
kernel_size = 3

"""
Input shape: 3D tensor with shape: (batch_size, steps, input_dim)
Output shape: 3D tensor with shape: (batch_size, new_steps, filters) 
steps value might have changed due to padding or strides.
"""

model.add(Convolution1D(filters=num_filters, kernel_size=kernel_size, input_shape=(B, N, F, T)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2))


sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["mae"])