import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD, Adagrad, Adam, Nadam
from keras.utils import np_utils
from keras import backend


from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LambdaCallback

T = 100   # Number of time steps
R = 15   # Number of players
F = 15   # Number of features
batch_size = 128


print("Shape of input should be (N x T x F x 2R) or", (T, F, 2*R))


# Load data
data = np.load("./2003-2019-data.pkl.npy")
scores = np.load("./2003-2019-outputs.pkl.npy")
print("data.shape: ", data.shape)


# Swap axis to make sure that the data is of shape (N x T x F x 2R)
data = np.swapaxes(data, 1, 3)
data = np.swapaxes(data, 1, 2)


print("swapaxes_data.shape: ", data.shape)
print("scores.shape: ", scores.shape)

num_filters = R*2
kernel_size = 5
drop_out_rate = 0.1


################################################################################################################
# 
# Most important part, need to try out different architecture

# # Create neural network 
model = Sequential()

model.add(Conv2D(filters=num_filters, kernel_size=(50,3), padding='same', data_format='channels_last', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(filters=num_filters, kernel_size=(20,3), padding='same', data_format='channels_last', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(padding='same'))


model.add(Conv2D(filters=num_filters, kernel_size=(10,3), padding='same', data_format='channels_last', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(filters=num_filters, kernel_size=(5,3), padding='same', data_format='channels_last', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(padding='same'))

model.add(Conv2D(filters=num_filters, kernel_size=(3,3), padding='same', data_format='channels_last', use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(MaxPooling2D(padding='same'))

model.add(Flatten())

model.add(Dropout(drop_out_rate))

model.add(Dense(2048, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(drop_out_rate))

model.add(Dense(1024, use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(drop_out_rate))

model.add(Dense(512,use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(drop_out_rate))

model.add(Dense(256,use_bias=False))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(drop_out_rate))


model.add(Dense(2))

################################################################################################################

# Optimizer
optimizer = Adam(lr=0.001)


# User-defined function to compute accuracy of game prediction
def accuracy(y_true, y_pred):
	# Predict the team with the higher score wins
	y_true_class = backend.argmax(y_true)
	y_pred_class = backend.argmax(y_pred)
	return backend.round(backend.mean(backend.equal(y_true_class, y_pred_class)) * 100)


# Compile the computational graph 
model.compile(loss='logcosh', optimizer=optimizer, metrics=["mae", accuracy])


# Setting how much of the total data is used for validation
validation_split = 0.1
n_epochs         = 30


# CheckPoint object to save the best performing model
# monitor is used to specify which metrics we want to focus on
checkpoint = ModelCheckpoint("./Checkpoint/best_model.pk", monitor='val_loss', save_best_only=True)

# logger to keep track of training metrics
logger     = CSVLogger("./Checkpoint/training.log", separator=',')

# Do fitting, returns a history object
history = model.fit(x = data, y = scores, batch_size=10, epochs=n_epochs, validation_split=validation_split, callbacks=[checkpoint, logger])
