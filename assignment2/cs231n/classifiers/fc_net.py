from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        pass
        # W1 = input_dim x hidden_dim
        W1 = np.random.normal(0,weight_scale, (input_dim,hidden_dim))
        b1 = np.zeros([1,hidden_dim])
        W2 = np.random.normal(0,weight_scale, (hidden_dim,num_classes))
        b2 = np.zeros([1,num_classes])
        
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        pass
        W1 = self.params['W1']
        b1 = self.params['b1']
        W2 = self.params['W2']
        b2 = self.params['b2']
        
        #print(X.shape)
        #print(np.ones([X.shape[0],1]).shape)
        #test = np.hstack( (X, np.ones([X.shape[0],1])) )
        #print(X.shape)
        X_mod = X.reshape( (X.shape[0],-1))
        X_mod = np.hstack((X_mod, np.ones([X.shape[0],1]))) # (sample, input_dim + 1)
        W1_mod = np.vstack((W1,b1)) # (input_dim + 1, dim2)
        
        X2 = np.dot(X_mod , W1_mod) #(sample, dim2)
        X2_RELU = np.maximum(X2,0)
        
        X2_mod = np.hstack( (X2_RELU, np.ones([X2_RELU.shape[0],1])) ) #(sample, dim2+1)
        W2_mod = np.vstack((W2,b2)) # (dim2+1, class)
        scores = np.dot( X2_mod ,W2_mod ) #(sample, class)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        #np.exp(scores
        scores_exp = np.exp(scores)
        
        probs = scores_exp/np.sum(scores_exp,axis=1,keepdims=True)
        correct_log_probs = -np.log(probs[range(X.shape[0]),y])
        
        data_loss = np.sum(correct_log_probs)/X.shape[0]      
        reg_loss = 0.5*self.reg*(np.sum(W1*W1) + np.sum(W2*W2))
        loss = data_loss + reg_loss
        
        reg_dW1 = self.reg * W1
        reg_dW2 = self.reg * W2       
        #dScores = (probs - 1)
        
        dScores = probs #(sample, class)
        dScores[range(X.shape[0]),y] -= 1
        dScores = dScores /X.shape[0] #(N,C)
        
        dX2 = np.dot(dScores, W2.T) #(sample, dim2)
        db2 = np.sum(dScores,axis=0) #(1, num_class)  
        dW2 = np.dot(X2.T,dScores) + reg_dW2 # (dim2,class)
        
        dX1_RELU = (X2 > 0) * dX2
        dX1 = np.dot(dX1_RELU,W1.T)
        dW1 = np.dot(X.reshape( (X.shape[0],-1)).T, dX1_RELU) + reg_dW1
        db1 = np.sum(dX1_RELU, axis=0) 
        
        grads['W2'] = dW2 
        grads['b2'] = db2
        grads['W1'] = dW1
        grads['b1'] = db1
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.bk_params = {}
        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        pass
#        self.params["W1"] = np.random.normal(0,weight_scale, (input_dim,hidden_dims[0]))
#        self.params["b1"] = np.zeros([1,hidden_dims[0]])
#        self.params["gamma1"] = np.ones([1,hidden_dims[0]])
#        self.params["beta1"] = np.ones([1,hidden_dims[0]])
        
        
#        for c in np.arange(self.num_layers-2):
#            self.params["W" + str(c+2)] = np.random.normal(0,weight_scale, (input_dim,hidden_dims[c]))
#            self.params["b" + str(c+2)] = np.zeros([1,hidden_dims[c]])
#            self.params["gamma" + str(c+2)] = np.ones([1,hidden_dims[c]])
#            self.params["beta" + str(c+2)] = np.ones([1,hidden_dims[c]])
            
     
        
        for c in np.arange(self.num_layers):
            if (c != self.num_layers-1) & ((self.normalization == 'batchnorm') | (self.normalization == 'layernorm') ):
                self.params["gamma" + str(c+1)] = np.ones([1,hidden_dims[c]])
                self.params["beta" + str(c+1)] = np.zeros([1,hidden_dims[c]])
            
            if c == 0:
                self.params["W" + str(c+1)] = np.random.normal(0,weight_scale, (input_dim,hidden_dims[c]))
                self.params["b" + str(c+1)] = np.zeros([1,hidden_dims[c]])
                
            elif c == self.num_layers-1:
                self.params["W" + str(c+1)] = np.random.normal(0,weight_scale, (hidden_dims[c-1],num_classes))
                self.params["b" + str(c+1)] = np.zeros([1,num_classes])
            else :
                self.params["W" + str(c+1)] = np.random.normal(0,weight_scale, (hidden_dims[c-1],hidden_dims[c]))
                self.params["b" + str(c+1)] = np.zeros([1,hidden_dims[c]])
              
            
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        pass

        self.bk_params['X_tmp0'] = X.reshape((X.shape[0],-1))
        self.bk_params['A0'] = np.hstack( (self.bk_params['X_tmp0'], np.ones([self.bk_params['X_tmp0'].shape[0],1]) )) 
        
        for layer in range(self.num_layers):
            if(layer == 0):
                self.bk_params['Wb1'] = np.vstack((self.params["W1"] ,self.params["b1"]))
                self.bk_params['Z1'] = np.dot(self.bk_params['A0'],self.bk_params['Wb1'])
            else:
                if self.normalization == 'batchnorm':
                    batch_out, self.bk_params["cache" + str(layer)] = batchnorm_forward(self.bk_params["Z"+str(layer)], self.params["gamma" + str(layer)], self.params["beta" + str(layer)], self.bn_params[layer-1])
                    self.bk_params["X_tmp"+str(layer)] = np.maximum(batch_out,0)
                elif self.normalization == 'layernorm' :
                    batch_out, self.bk_params["cache" + str(layer)] = layernorm_forward(self.bk_params["Z"+str(layer)], self.params["gamma" + str(layer)], self.params["beta" + str(layer)], self.bn_params[layer-1])
                    self.bk_params["X_tmp"+str(layer)] = np.maximum(batch_out,0)
                    
                else :
                    self.bk_params["X_tmp"+str(layer)] = np.maximum(self.bk_params["Z"+str(layer)],0) 
                
                if self.use_dropout == 1: 
                    self.bk_params["X_tmp"+str(layer)], self.dropout_param["cache" + str(layer)]  = dropout_forward(self.bk_params["X_tmp"+str(layer)], self.dropout_param)
                
                self.bk_params["A"+str(layer)] = np.hstack( (self.bk_params["X_tmp"+str(layer)], np.ones([self.bk_params["X_tmp"+str(layer)].shape[0],1])))   
                self.bk_params["Wb"+str(layer+1)] = np.vstack((self.params["W"+str(layer+1)] ,self.params["b"+str(layer+1)]))
                self.bk_params["Z"+str(layer+1)] = np.dot(self.bk_params["A"+str(layer)], self.bk_params["Wb"+str(layer+1)] )
        scores = self.bk_params["Z"+str(self.num_layers)]                
         
                
            
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
    
        Weight_tmp = 0;
        
        scores_exp = np.exp(scores)        
        probs = scores_exp/np.sum(scores_exp,axis=1,keepdims=True)
        correct_log_probs = -np.log(probs[range(X.shape[0]),y])
        
        data_loss = np.sum(correct_log_probs)/X.shape[0]    
        
        for layer in range(self.num_layers):
            self.bk_params["reg_dW"+str(layer+1)] = np.sum(self.params["W"+str(layer+1)] * self.params["W"+str(layer+1)])
            Weight_tmp = Weight_tmp + self.bk_params["reg_dW"+str(layer+1)]
        
        reg_loss = 0.5*self.reg*Weight_tmp
        loss = data_loss + reg_loss
        
        #reg_dW1 = self.reg * W1
        #reg_dW2 = self.reg * W2       
        #dScores = (probs - 1)
        
        dScores = probs
        dScores[range(X.shape[0]),y] -= 1
        dScores = dScores /X.shape[0] #(N,C)
        
        #The last layer
        self.bk_params["dX" + str(self.num_layers)] = np.dot(dScores, self.params["W"+str(self.num_layers)].T)
        self.bk_params["db" + str(self.num_layers)] = np.sum(dScores,axis=0)
        self.bk_params["dW" + str(self.num_layers)] = np.dot(self.bk_params["X_tmp"+str(self.num_layers-1)].T,dScores) + self.reg * self.params["W"+str(self.num_layers)]                          
        
        grads['W'+str(self.num_layers)] = self.bk_params["dW"+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = self.bk_params["db"+str(self.num_layers)]
 

     
        for layer in reversed(range(self.num_layers-1)):
            #print(layer)
            #print(self.bk_params)
            #print(self.bk_params('X_tmp0'))
            if self.use_dropout == 1:
                self.bk_params["dX" + str(layer+1) + "_RELU"] = dropout_backward((self.bk_params["X_tmp"+str(layer+1)] > 0) * self.bk_params["dX"+str(layer+2)],self.dropout_param["cache" + str(layer+1)] )
            else :
                self.bk_params["dX" + str(layer+1) + "_RELU"] = (self.bk_params["X_tmp"+str(layer+1)] > 0) * self.bk_params["dX"+str(layer+2)]
                
            if self.normalization == 'batchnorm':
                self.bk_params["dX" + str(layer+1) + "_RELU"], grads['gamma'+str(layer+1)], grads['beta'+str(layer+1)] = batchnorm_backward(self.bk_params["dX" + str(layer+1) + "_RELU"], self.bk_params["cache" + str(layer+1)])
            elif self.normalization == 'layernorm' :
                self.bk_params["dX" + str(layer+1) + "_RELU"], grads['gamma'+str(layer+1)], grads['beta'+str(layer+1)] = layernorm_backward(self.bk_params["dX" + str(layer+1) + "_RELU"], self.bk_params["cache" + str(layer+1)])
            self.bk_params["dX" + str(layer+1)] = np.dot(self.bk_params["dX" + str(layer+1) + "_RELU"], self.params["W"+ str(layer+1)].T)
            self.bk_params["dW" + str(layer+1)] = np.dot(self.bk_params[("X_tmp"+str(layer))].reshape( (self.bk_params[("X_tmp"+str(layer))].shape[0],-1)).T, self.bk_params[("dX"+str(layer+1)+"_RELU")]) + self.reg * self.params["W"+str(layer+1)]
            self.bk_params["db" + str(layer+1)] = np.sum(self.bk_params["dX" + str(layer+1) + "_RELU"],axis=0)
                           
            grads['W'+str(layer+1)] = self.bk_params["dW"+str(layer+1)]
            grads['b'+str(layer+1)] = self.bk_params["db"+str(layer+1)]

        #dX1_RELU = (X2 > 0) * dX2
        #dX1 = np.dot(dX1_RELU,W1.T)
        #dW1 = np.dot(X.reshape( (X.shape[0],-1)).T, dX1_RELU) + reg_dW1
        #db1 = np.sum(dX1_RELU, axis=0) 
        
        #grads['W2'] = dW2 
        #grads['b2'] = db2
        #grads['W1'] = dW1
        #grads['b1'] = db1
        
    
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
