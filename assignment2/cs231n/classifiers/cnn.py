from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        C,H,W = input_dim

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        pass
        #print(input_dim) 
        #print(hidden_dim)
        #(F,C,HH,WW) (F)
        # W1 ( num_filters, C, H', W')
        
        
        self.params['W1'] = np.random.normal(0,weight_scale, (num_filters,input_dim[0],filter_size,filter_size))
        self.params['b1'] = np.zeros([num_filters])
        self.params['W2'] = np.random.normal(0,weight_scale, (int(num_filters*input_dim[1]/2*input_dim[2]/2),hidden_dim))
        self.params['b2'] = np.zeros([hidden_dim])
        self.params['W3'] = np.random.normal(0,weight_scale, (hidden_dim,num_classes))
        self.params['b3'] = np.zeros([num_classes])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        # conv - relu - 2x2 max pool - affine - relu - affine - softmax
              
        N, C, H, W = X.shape
        #F, _, HH, WW = w.shape
        #_, _, out_h, out_w = dout.shape
        
        X1, cache = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        A1 = np.reshape(X1,(N,-1)) # Is this reshaping correct?
        X1_mod = np.hstack( (A1,np.ones([A1.shape[0],1]))) # (N, +1)
        W2_mod = np.vstack((W2,b2))
        X2 = np.dot(X1_mod,W2_mod)   # (N, hidden_dim)
        RELU_X2 = np.maximum(X2,0)
        mask_X2 = X2 > 0
        X2_mod = np.hstack((RELU_X2,np.ones([RELU_X2.shape[0],1])))
        W3_mod = np.vstack((W3,b3))
        Z3 = np.dot(X2_mod,W3_mod) 
        scores = Z3.copy()
        A3 = Z3.copy()
        A3 -= np.max(A3)

        P = np.exp(A3)/np.sum(np.exp(A3),axis=1,keepdims=True)
        
        correct_log_P = -np.log(P[range(N),y])
        data_loss = np.sum(correct_log_P)/N
        reg_loss = 0.5 *self.reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
        
        
        #print(np.log(W3.shape[1]))
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = data_loss + reg_loss, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        #grads['W1'] = np.random.normal(0,weight_scale, (num_filters,input_dim[0],filter_size,filter_size))
        #grads['b1'] = np.zeros([num_filters])
        #grads['W2'] = np.random.normal(0,weight_scale, (int(num_filters*input_dim[1]/2*input_dim[2]/2),hidden_dim))
        #grads['b2'] = np.zeros([hidden_dim])
        #grads['W3'] = np.random.normal(0,weight_scale, (hidden_dim,num_classes))
        #grads['b3'] = np.zeros([num_classes])
        
        dScores = P
        dScores[range(N),y] -= 1
        dScores = dScores/N
        
        reg_dW3 = self.reg * W3
        reg_dW2 = self.reg * W2
        #grads['X3'] = np.dot(dScores, W3.T)
        grads['W3'] = np.dot(RELU_X2.T,dScores) + reg_dW3
        grads['b3'] = np.sum(dScores,axis=0)
        grads_X3 = np.dot(dScores, W3.T) * mask_X2 #(N, hidden)
        grads['b2'] = np.sum(grads_X3,axis=0)
        grads['W2'] = np.dot(A1.T, grads_X3) + reg_dW2
        grads_X2 = np.dot(grads_X3, W2.T)
        grads_X1,grads['W1'],grads['b1'] = conv_relu_pool_backward(np.reshape(grads_X2,X1.shape),cache)
              
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads