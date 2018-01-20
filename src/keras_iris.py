##################################
# IMPORTS
##################################
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from time import time as time
import arff

##################################
# CONFIGURATION
##################################
numberOfHiddenLayers = 6
denseLayerNeurons = [100, 30, 30, 80, 10]
activationFunctionsHidden = ['relu', 'relu', 'relu', 'relu', 'relu']
dropoutLayerPositions = [1] #0-based indices, where 0 is not allowed
dropoutValues = [0.2]
activationFunctionOutput = 'softmax'
batchSize = 1000
epochs = 60

##################################
# LOAD DATA
##################################
training_dataframe = arff.load(open('../data/forest_fire.arff'))      #<=====================
training_data = numpy.array(training_dataframe['data'])
test_dataframe = arff.load(open('../data/forest_fire_test.arff'))     #<=====================
test_data = numpy.array(test_dataframe['data'])

# get number of classes and dimensions
with open('../data/forest_fire_test.arff') as fh:        #<=====================
    arffData = arff.ArffDecoder().decode(fh)
    dataAttributes = dict(arffData['attributes'])
classes = len(dataAttributes['class'])
dims = len(dataAttributes)-1

X_train = training_data[:, 0:dims].astype(float)
Y_train = training_data[:, dims]
X_test = test_data[:, 0:dims].astype(float)
Y_test = test_data[:, dims]

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
denseCounter = 0
dropoutCounter = 0
model = Sequential()
for i in range(0, numberOfHiddenLayers):
    if i == 0:
        model.add(
            Dense(denseLayerNeurons[denseCounter], input_dim=dims, activation=activationFunctionsHidden[denseCounter]))
        denseCounter += 1
    elif dropoutCounter < len(dropoutLayerPositions) & i == dropoutLayerPositions[dropoutCounter]:
        model.add(Dropout(dropoutValues[dropoutCounter]))
        dropoutCounter += 1
    elif denseCounter < len(denseLayerNeurons):
        model.add(
            Dense(denseLayerNeurons[denseCounter], activation=activationFunctionsHidden[denseCounter]))
        denseCounter += 1

model.add(Dense(classes, activation=activationFunctionOutput))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# train model
t = time()
history = model.fit(X_train, dummy_y_train, batch_size=batchSize, epochs=epochs, verbose=1)
training_time = time() - t


##################################
# TESTING
##################################
t = time()
test_eval = model.evaluate(X_test, dummy_y_test, verbose=1)
test_time = time() - t
print(test_eval)


# plot accuracy over epochs
line_acc, = pyplot.plot(history.history['acc'], label='accuracy')
line_loss, = pyplot.plot(history.history['loss'], label='loss')
pyplot.legend(handles=[line_acc, line_loss])
pyplot.xlabel('epochs')
pyplot.show()
