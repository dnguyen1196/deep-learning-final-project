import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils


# ########################################
#  Load training and testing data
# ########################################
# TODO: replace this with NBA train and test files
train = pd.read_csv('./example_train.csv')
test = pd.read_csv('./example_test.csv')

########################################
#  Helper functions do data transformation? If needed
########################################
def encode(train, test):
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)
    train = train.drop(['species', 'id'], axis=1)
    test = test.drop('id', axis=1)
    return train, labels, test, classes
train, labels, test, classes = encode(train, test)


# ########################################
#
#			PREPROCESSING			
#
# ########################################
# Scale train features, preprocessing step
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

# split train data into train and validation
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]


nb_features = 64 # number of features per features type (shape, texture, margin)   
nb_class = len(classes)

# Convert to categorical value (interesting helper function, not necessarily will be used)
y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

# ########################################
#
# ########################################

# Note that in this particular example, the train and validataion tensor is
# of shape (num_examples, num_features, 3) 
# The following example basically constructs a 3D tensor from data matrix

# reshape train data
X_train_r = np.zeros((len(X_train), nb_features, 3))
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:128]
X_train_r[:, :, 2] = X_train[:, 128:]

# reshape validation data
X_valid_r = np.zeros((len(X_valid), nb_features, 3))
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
X_valid_r[:, :, 2] = X_valid[:, 128:]



# Create neural network
model = Sequential()
model.add(Convolution1D(filters=512, kernel_size=1, input_shape=(nb_features, 3)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class))
model.add(Activation('softmax'))


sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)

# TODO: modify the loss function to binary classification (if we are predicting WIN-LOSS) 
# TODO: loss = 'mean_squared_error' when we are trying to predict scores
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Training
nb_epoch = 15
model.fit(X_train_r, y_train, epochs=nb_epoch, validation_data=(X_valid_r, y_valid), batch_size=16)



