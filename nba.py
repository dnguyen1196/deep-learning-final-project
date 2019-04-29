import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Conv1D, Dropout, Conv2D
from keras.optimizers import SGD
from keras.utils import np_utils

T = 10   # Number of time steps
R = 15   # Number of players
F = 11  # Number of features
total_points = 128 # Batch size


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


print("Shape of input should be (T x F x 2R) or", (T, F, 2*R))
print("Shape of input is ", X_batch[0].shape)

X_batch = np.array(X_batch)
Y_batch = np.array(Y_batch)

# Concatenation to form input
# Create neural network
model = Sequential()




from keras import backend as K
from keras.layers import Layer

class MyConv1D(Layer):
	def __init__(self, input_shape, filters, kernel_size, stride, padding, **kwargs):
		self.input_shape = input_shape
		self.filters = filters
		self.kernel_size = kernel_size
		self.stride = stride
		self.padding = padding
		super(MyConv1D, self).__init__(**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.kernel = self.add_weight(name='kernel', 
									  shape=(input_shape[1], self.output_dim),
									  initializer='uniform',
									  trainable=True)

		self.filters_list = []

		H, W, D = input_shape[0], input_shape[1], input_shape[2]
		for num in range(self.filters):
			weight = self.add_weight(name='filter_{}'.format(num), shape=(self.kernel_size, ))

		super(MyConv1D, self).build(input_shape)  # Be sure to call this at the end

	def call(self, x):
		(batch_size, T, F, R_times_2) = x.shape
		return K.dot(x, self.kernel)

	def compute_output_shape(self, input_shape):
		assert(input_shape.shape == (3, ))
		H, W, D = input_shape[0], input_shape[1], input_shape[2]
		new_H = int(H + 2 * padding - self.kernel_size)/ self.stride + 1
		new_W = self.filters
		self.output_shape = (new_H, new_W, D)
		return (new_H, new_W, D)


suggested_indices_of_interest = ['game_score',\
								 'made_three_point_field_goals', 'attempted_three_point_field_goals', \
								 'made_field_goals', 'attempted_field_goals', \
								 'made_free_throws', 'attempted_free_throws', \
								 'assists', 'offensive_rebounds',\
								 'seconds_played', \
								 'defensive_rebounds', 'turnovers',\
								 'steals',  'blocks', 'personal_fouls']


num_filters = R*2
kernel_size = 5

model.add(Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), input_shape=(T, F, 2*R)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(2))


sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='mean_squared_error', optimizer=sgd, metrics=["mae"])
model.fit(x = X_batch, y = Y_batch, batch_size=10)
