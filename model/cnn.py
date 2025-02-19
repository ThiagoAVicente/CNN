import numpy as np

from functions.relu_derivate import *
from functions.con2d import *
from functions.relu import *
from functions.max_pooling import *
from functions.softmax import *
from functions.dense import *

eps:float = 1e-20

class cnn:

    def __init__(self, input_size:int, num_of_filters:int , filter_size:int ,
                    num_classes:int , learning_rate:float = 0.01) -> None:

        # Initialize parameters
        self.num_of_filters:int = num_of_filters
        self.filter_size:int = filter_size
        self.filters:np.ndarray = np.random.randn(num_of_filters, filter_size, filter_size)

        self.output_height:int = input_size - filter_size + 1
        self.output_width:int = input_size - filter_size + 1
        self.flattened_size:int = self.output_height * self.output_width * num_of_filters

        self.weights:np.ndarray = np.random.randn(self.flattened_size, num_classes)
        self.bias:np.ndarray = np.zeros((1, num_classes))
        self.learning_rate:float = learning_rate

        #print(self.weights.shape)
        #print(self.bias.shape)

    def forward_propagation(self, input_image:np.ndarray )-> np.ndarray:
        """
        computes the model prediction
        :param input_image -> the input layer image
        :return -> the output_layer of the model
        """

        self.input_image = input_image

        """
        Convutional
        -> search for features using the filters
        """
        features = []
        for i in range ( self.num_of_filters ):
            o = con2d( input_image, self.filters[i] )
            features.append(o)
        self.features = np.array( features )

        """
        Apply activation function
        """
        self.relu_features = relu( self.features );

        """
        faltten data for future use
        """
        self.flatten_features = self.relu_features.flatten().reshape( [1,-1] )

        """
        calculate the output of the dense layer
        """
        self.dense_output = dense( self.flatten_features, self.weights, self.bias )

        """
        Use softmax to get the probabilities
        """
        self.output = softmax(self.dense_output)
        return self.output

    def backward_propagation(self, true_labels:np.ndarray):
        """
        Change the model parameters based on the last prediction error
        :param true_labels -> one-hot-encoded
        """

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

        # UPDATE
        self.filters -= self.learning_rate * grad_filters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db

        return dw, db, grad_filters

    def train(self, train_data:np.ndarray, train_labels:np.ndarray, limit:int = 100, batch_size:int = 1,
        thereshold:float = 0.5):
        """
        train the model with the given data
        :param limit -> max number of iteractions
        :batch_size -> size of each trainning batch
        """

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

            # thereshold
            if iter_loss < thereshold:
                break
