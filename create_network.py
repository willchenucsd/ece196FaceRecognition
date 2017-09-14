"""
ECE196 Face Recognition Project
Author: W Chen

Adapted from: https://keras.io/getting-started/functional-api-guide/

Modify this code to write a LeNet with the following requirements:
* Input dimensions: 32x32x1
* C1: convolutional layer, output: 6 layers of 28x28 feature maps, filter size: 5x5,
  strides: 1 both horizontally and vertically, activation function: sigmoid
* S2: max pooling layer, output: 6 layers of 14x14 feature maps, pooling size: 2x2,
  strides: 2 both horizontally and vertically
* C3: convolutional layer, output: 16 layers of 28x28 feature maps, filter size: 5x5,
  strides: 1 both horizontally and vertically, activation function: sigmoid
* S4: max pooling layer, output: 16 layers of 28x28 feature maps, pooling size: 2x2,
  strides: 2 both horizontally and vertically
* C5: convolutional layer, output: 120 layers of 1x1 feature maps, filter size: 5x5, activation function: sigmoid
* F6: fully connected layer, output 84-dimensional vector, activation function: tanh
* F7: fully connected layer, output 10-dimensional vector, activation function: softmax

"""

from keras.layers import Input, Dense
from keras.models import Model

# This returns a tensor
inputs = Input(shape=(784,))

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)

# print model architecture
model.summary()

