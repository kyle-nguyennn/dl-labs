""" 			  		 			     			  	   		   	  			  	
Softmax Regression Model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

# Do not use packages that are not in standard distribution of python
import numpy as np

from ._base_network import _baseNetwork


class SoftmaxRegression(_baseNetwork):
    def __init__(self, input_size=28 * 28, num_classes=10):
        """
        A single layer softmax regression. The network is composed by:
        a linear layer without bias => (activation) => Softmax
        :param input_size: the input dimension
        :param num_classes: the number of classes in total
        """
        super().__init__(input_size, num_classes)
        self._weight_init()

    def _weight_init(self):
        '''
        initialize weights of the single layer regression network. No bias term included.
        :return: None; self.weights is filled based on method
        - W1: The weight matrix of the linear layer of shape (num_features, hidden_size)
        '''
        np.random.seed(1024)
        self.weights['W1'] = 0.001 * np.random.randn(self.input_size, self.num_classes)
        self.gradients['W1'] = np.zeros((self.input_size, self.num_classes))

    def forward(self, X, y, mode='train'):
        """
        Compute loss and gradients using softmax with vectorization.

        :param X: a batch of image (N, 28x28)
        :param y: labels of images in the batch (N,)
        :return:
            loss: the loss associated with the batch
            accuracy: the accuracy of the batch
        """
        loss = None
        gradient = None
        accuracy = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the forward process and compute the Cross-Entropy loss    #
        #    2) Compute the gradient of the loss with respect to the weights        #
        # Hint:                                                                     #
        #   Store your intermediate outputs before ReLU for backwards               #
        #############################################################################

        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        W = self.weights['W1']
        n = y.shape[0]
        f = W.shape[0]
        print(f"N = {n}")
        out = np.matmul(X, W)
        a = self.ReLU(out)
        _Dout = self.ReLU_dev(out) # local gradient of a to out
        h = self.softmax(a)
        loss = self.cross_entropy_loss(h, y)
        accuracy = self.compute_accuracy(h, y)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        if mode != 'train':
            return loss, accuracy

        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the backward process:                                     #
        #        1) Compute gradients of each weight by chain rule                  #
        #        2) Store the gradients in self.gradients                           #
        #############################################################################
        y_onehot = self._onehot(y, self.num_classes)
        a_ = y_onehot # uptream backward gradient at a
        _DW = np.broadcast_to((X.T.sum(axis=1)/n).reshape((f, 1)), (f, self.num_classes))
        print(f"shape of a_: {a_.shape}")
        print(f"shape of _Dout: {_Dout.shape}")
        print(f"shape of _DW.T: {_DW.T.shape}")
        grad = np.multiply(a_, _Dout) @ (_DW.T)
        print(f"Upstream grad a W: {grad}")
        self.gradients['W1'] = np.broadcast_to((grad.sum(axis=0)/n).reshape((1, f)), (self.num_classes, f)).T

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss, accuracy
