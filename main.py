import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, Conv2D
from keras.optimizers import SGD, Adagrad, Adam
from keras.utils import np_utils
from keras import backend


from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LambdaCallback

T = 10   # Number of time steps
R = 15   # Number of players
F = 15  # Number of features
total_points = 128 # Batch size
batch_size = 128

# Create a fake batch
X_batch = []
Y_batch = [] 

for _ in range(total_points):
	home_team = np.random.rand(R, F, T)
	away_team = np.random.rand(R, F, T)

	# Need to rotate tensor into shape (F, T, R) or (T, F, R)
	home_team = np.swapaxes(home_team, 0, 2)
	away_team = np.swapaxes(away_team, 0, 2)

	y1        = np.random.randint(0, 100)
	y2        = np.random.randint(0, 100)
	X         = np.concatenate((home_team, away_team), axis=2)

	X_batch.append(X)
	Y_batch.append([y1, y2])

# print("Shape of input is ", X_batch[0].shape)
# X_batch = np.array(X_batch)
# Y_batch = np.array(Y_batch)

print("Shape of input should be (N x T x F x 2R) or", (T, F, 2*R))


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

# # Create neural network
model = Sequential()
model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), input_shape=(T, F, 2*R)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2))


# sgd = SGD(lr=0.0001, nesterov=True, decay=1e-6, momentum=0.9)
optimizer = Adagrad(lr=0.0001, decay=0)


def accuracy(y_true, y_pred):
	y_true_class = backend.argmax(y_true)
	y_pred_class = backend.argmax(y_pred)
	return backend.mean(backend.equal(y_true_class, y_pred_class))


# Compile the computational graph 
model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=["mae", accuracy])

# Do fitting
"""
fit(x=None, y=None, batch_size=None, epochs=1, verbose=1, 
 				callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, 
 				class_weight=None, sample_weight=None, initial_epoch=0, steps_per_epoch=None, 
 				validation_steps=None, validation_freq=1)


callbacks: List of keras.callbacks.Callback instances. 
		List of callbacks to apply during training and validation (if ). See callbacks.

validation_split = 


"""
validation_split = 0.5


checkpoint = ModelCheckpoint("./Checkpoint/best_model.pk", monitor='val_loss', save_best_only=True)
logger     = CSVLogger("./Checkpoint/training.log", separator=',')

model.fit(x = data, y = scores, batch_size=10, epochs=10, validation_split=validation_split, callbacks=[checkpoint, logger])
