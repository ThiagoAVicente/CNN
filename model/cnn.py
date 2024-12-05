import numpy as np

from functions.relu_derivate import *
from functions.con2d import *
from functions.relu import *
from functions.max_pooling import *
from functions.softmax import *
from functions.dense import *

eps = 1e-20
class cnn:

    def __init__(self, input_size, num_of_filters , filter_size, num_classes, learning_rate = 0.01):

        # Initialize parameters
        self.num_of_filters = num_of_filters
        self.filter_size = filter_size
        self.filters = np.random.randn(num_of_filters, filter_size, filter_size)

        self.output_height = input_size - filter_size + 1
        self.output_width = input_size - filter_size + 1
        self.flattened_size = self.output_height * self.output_width * num_of_filters

        self.weights = np.random.randn(self.flattened_size, num_classes)
        self.bias = np.zeros((1, num_classes))
        self.learning_rate = learning_rate

        #print(self.weights.shape)
        #print(self.bias.shape)

    def forward_propagation(self, input_image ):

        self.input_image = input_image

        #Convutional
        features = []
        for i in range ( self.num_of_filters ):
            o = con2d( input_image, self.filters[i] )
            features.append(o)
        self.features = np.array( features )

        #Relu activation
        self.relu_features = relu( self.features );

        #Flatten
        self.flatten_features = self.relu_features.flatten().reshape( [1,-1] )
        #dense layer
        self.dense_output = dense( self.flatten_features, self.weights, self.bias )

        #softmax
        self.output = softmax(self.dense_output)
        return self.output

    def backward_propagation(self, true_labels):

        # Number of samples
        m = true_labels.shape[0]
        output_loss = self.output - true_labels;

        # DENSE LAYER
        dz = output_loss
        dw = np.dot( self.flatten_features.T, dz ) / m
        db = np.sum(dz,axis=0,keepdims=True) / m

        # RELU
        dz_relu = np.dot( dz, self.weights.T )

        dz_relu = dz_relu.reshape(self.features.shape)

        dz_relu *= relu_derivative(self.features)

        # CONVUTIONAL
        grad_filters = np.zeros_like(self.filters)
        for i in range(self.num_of_filters):
            for j in range(self.features.shape[1]):
                for k in range(self.features.shape[2]):
                    region = self.input_image[j:j+self.filter_size, k:k+self.filter_size]
                    grad_filters[i] += region * dz_relu[i, j, k]

        # ::UPDATE::
        self.filters -= self.learning_rate * grad_filters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return dw, db, grad_filters

    def train(self, train_data, train_labels, limit = 100, batch_size = 1 ):

        num_samples = train_data.shape[0]

        for iter in range ( limit ):
            iter_loss = 0

            for j in range( 0, num_samples, batch_size ):

                ## DATA
                batch_data = train_data[j: j + batch_size]
                batch_labels = train_labels[j: j + batch_size]

                for k in range( batch_data.shape[0] ):

                    input_image = batch_data[k]
                    label = batch_labels[k]
                    self.forward_propagation(input_image)

                    # CROSS-ENTROPY LOSS
                    loss = -np.sum( label * np.log( self.output + eps ) )
                    iter_loss += loss

                    self.backward_propagation(label)

            print(f"Iteration {iter+1}, loss: {iter_loss/num_samples} ")

            if iter_loss < 0.5:
                break
