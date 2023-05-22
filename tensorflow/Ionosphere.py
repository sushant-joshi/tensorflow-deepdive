import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.initializers import GlorotUniform, HeUniform

# dataset definition
class CSVDataset:
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = pd.read_csv(path, header=None)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self, n_test=0.33):
        # determine sizes
        test_size = round(n_test * len(self.X))
        train_size = len(self.X) - test_size
        # calculate the split
        return np.random.permutation(len(self.X))[:train_size], np.random.permutation(len(self.X))[train_size:]

# model definition
class MLP(models.Model):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        self.hidden1 = layers.Dense(10, activation='relu', kernel_initializer=GlorotUniform())
        # second hidden layer
        self.hidden2 = layers.Dense(8, activation='relu', kernel_initializer=GlorotUniform())
        # third hidden layer and output
        self.hidden3 = layers.Dense(1, activation='sigmoid', kernel_initializer=HeUniform())

    # forward propagate input
    def call(self, inputs):
        # input to first hidden layer
        x = self.hidden1(inputs)
        # second hidden layer
        x = self.hidden2(x)
        # third hidden layer and output
        x = self.hidden3(x)
        return x

# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    train_idx, test_idx = dataset.get_splits()
    # prepare data
    train_X, train_y = dataset.X[train_idx], dataset.y[train_idx]
    test_X, test_y = dataset.X[test_idx], dataset.y[test_idx]
    return train_X, train_y, test_X, test_y

# train the model
def train_model(train_X, train_y, model):
    # define the optimization
    criterion = BinaryCrossentropy()
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    # compile the model
    model.compile(optimizer=optimizer, loss=criterion)
    # train the model
    model.fit(train_X, train_y, epochs=100, batch_size=32, verbose=0)

# evaluate the model
def evaluate_model(test_X, test_y, model):
    # make predictions
    yhat = model.predict(test_X)
    # round predictions to 0 or 1
    yhat = np.round(yhat)
    # calculate accuracy
    acc = accuracy_score(test_y, yhat)
    return acc

# make a class prediction for one row of data
def predict(row, model):
    # make prediction
    yhat = model.predict(np.array([row]))
    # round prediction to 0 or 1
    yhat = np.round(yhat)
    return yhat

# prepare the data
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
train_X, train_y, test_X, test_y = prepare_data(path)
print(len(train_X), len(test_X))
# define the network
model = MLP(34)
# train the model
train_model(train_X, train_y, model)
# evaluate the model
acc = evaluate_model(test_X, test_y, model)
print('Accuracy: %.3f' % acc)
# make a single prediction (expect class=1)
row = [1,0,0.99539,-0.05889,0.85243,0.02306,0.83398,-0.37708,1,0.03760,0.85243,-0.17755,0.59755,-0.44945,0.60536,-0.38223,0.84356,-0.38542,0.58212,-0.32192,0.56971,-0.29674,0.36946,-0.47357,0.56811,-0.51171,0.41078,-0.46168,0.21266,-0.34090,0.42267,-0.54487,0.18641,-0.45300]
yhat = predict(row, model)
print('Predicted: %.3f (class=%d)' % (yhat, yhat))
