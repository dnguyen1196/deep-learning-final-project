import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils
from keras import backend


from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LambdaCallback

T = 10   # Number of time steps
R = 15   # Number of players
F = 15   # Number of features
batch_size = 128


print("Shape of input should be (N x T x F x 2R) or", (T, F, 2*R))


# Load data
data = np.load("./data.pkl.npy")
scores = np.load("./outputs.pkl.npy")
print("data.shape: ", data.shape)


# Swap axis to make sure that the data is of shape (N x T x F x 2R)
data = np.swapaxes(data, 1, 3)
data = np.swapaxes(data, 1, 2)

data = np.concatenate((data, data), axis=0)
scores = np.concatenate((scores, scores), axis=0)

print("swapaxes_data.shape: ", data.shape)
print("scores.shape: ", scores.shape)

num_filters = R*2
kernel_size = 5


################################################################################################################
# 
# Most important part, need to try out different architecture

# # Create neural network 
model = Sequential()

# Convolution sequence
# model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), input_shape=(T, F, 2*R)))
# # model.add(Activation('relu'))
# model.add(MaxPooling2D())
# model.add(Conv2D(filters=num_filters, kernel_size=2))
# model.add(MaxPooling2D())

# Fully connected layer
model.add(Flatten())
# model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
# model.add(Dropout(0.1))
# model.add(Dense(1024, activation='relu'))
model.add(Dense(2))


################################################################################################################

# Optimizer
# sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
optimizer = Adagrad(lr=0.0001, decay=0)


# User-defined function to compute accuracy of game prediction
def accuracy(y_true, y_pred):
	# Predict the team with the higher score wins
	y_true_class = backend.argmax(y_true)
	y_pred_class = backend.argmax(y_pred)
	return backend.round(backend.mean(backend.equal(y_true_class, y_pred_class)) * 100)


# Compile the computational graph 
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mae", accuracy])


# Setting how much of the total data is used for validation
validation_split = 0.5
n_epochs         = 30


# CheckPoint object to save the best performing model
# monitor is used to specify which metrics we want to focus on
checkpoint = ModelCheckpoint("./Checkpoint/best_model.pk", monitor='val_loss', save_best_only=True)

# logger to keep track of training metrics
logger     = CSVLogger("./Checkpoint/training.log", separator=',')

# Do fitting, returns a history object
history = model.fit(x = data, y = scores, batch_size=10, epochs=n_epochs, validation_split=validation_split, callbacks=[checkpoint, logger])
