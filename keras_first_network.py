# Neural Network project (1)

# We will use the NumPy library to load our dataset and we will use 
# two classes from the Keras library to define our model
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
# The model expects rows of data with 8 variables (the input_dim=8 argument)

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # The first hidden layer has 12 nodes and uses the relu activation function.
model.add(Dense(8, activation='relu')) # The second hidden layer has 8 nodes and uses the relu activation function.
model.add(Dense(1, activation='sigmoid')) # The output layer has one node and uses the sigmoid activation function.

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make probability predictions with the model
# predictions = model.predict(X)

# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)

# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))