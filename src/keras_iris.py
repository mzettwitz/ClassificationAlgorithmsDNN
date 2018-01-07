##################################
#imports
##################################
import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from time import time as time
import arff

##################################
# LOAD DATA
##################################
training_dataframe = arff.load(open('../data/fish.arff'))
training_data = numpy.array(training_dataframe['data'])
test_dataframe = arff.load(open('../data/fish_test.arff'))
test_data = numpy.array(test_dataframe['data'])

X_train = training_data[:,0:463].astype(float)
Y_train = training_data[:,463]
X_test = test_data[:,0:463].astype(float)
Y_test = test_data[:,463]

##################################
# SETUP
##################################
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y_train = encoder.transform(Y_train)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_train = np_utils.to_categorical(encoded_Y_train)

encoder_test = LabelEncoder()
encoder_test.fit(Y_test)
encoded_Y_test = encoder_test.transform(Y_test)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y_test = np_utils.to_categorical(encoded_Y_test)

##################################
# BUILD DNN
##################################
# create model
model = Sequential()
model.add(Dense(512, input_dim=463, activation='relu'))
model.add(Dense(7, activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
t = time()
history = model.fit(X_train, dummy_y_train, batch_size=40, epochs=128, verbose=1)
training_time = time() - t

# plot accuracy over epochs
pyplot.plot(history.history['acc'])
pyplot.show()

##################################
# TESTING
##################################
t = time()
test_eval = model.evaluate(X_test, dummy_y_test, verbose=1)
test_time = time() - t
print(test_eval)


