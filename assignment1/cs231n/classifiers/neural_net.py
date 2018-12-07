
from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # X = (N,D)
    #self.params['W1'] = D, H
    #self.params['b1'] = H
    #self.params['W2'] = H,C
    #self.params['b2'] = C
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    N, D = X.shape
    
   
    
    W1_tmp = W1
    W2_tmp = W2
    # Compute the forward pass
    scores = None
    #print(X.shape)
    #print(np.ones( (X.shape[0],1) ).shape)
    #X = X - np.mean(X, axis = 0,keepdims=True)
    X_tmp = X
    X = np.hstack( (X,np.ones( (X.shape[0],1) ))) #(N,D+1)
    W1 = np.vstack( (W1,np.reshape(b1, (1,-1) ))) #(D+1,H)
    X2 = np.dot(X,W1) #(N,H)
    X2 = np.maximum(0,X2)    
    
    #U2 = (np.random.rand(*X2.shape) < 0.5)/0.5
    #X2 *= U2
    
    X2_tmp = X2
    #X2 = np.exp(X2 - np.max(X2,axis=1,keepdims=True))
    #X2 = X2/np.sum(X2,axis=1,keepdims=True)
    #X2 = -np.log(X2)
    
    X2 = np.hstack( (X2,np.ones( (X2.shape[0],1) ))) #(N,H+1)   
    
    W2 = np.vstack((W2,b2)) # (H+1,C)
    
    scores = np.dot(X2,W2) #(N,C)
    #exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
    #probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)

    #scores = -np.log(probs)
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
    pass

    #X2 = np.dot(X,W1) #(N,H)
    #X2 = np.exp(X2)
    #X2 = X2/np.sum(X2,axis=1,keepdims=True)
    #X2 = -np.log(X2)
    #X2 = np.hstack( (X2,np.ones( (X2.shape[0],1) ))) #(N,H+1)   
    #scores = np.dot(X2,W2) #(N,C)
    #exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
    #probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)

    

    exp_scores = np.exp(scores - np.max(scores,axis=1,keepdims=True))
    #exp_scores = np.exp(scores)   
    probs = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    real_prob = np.sum(-np.log(probs[range(N),y]))
    #real_prob = np.sum(probs[range(N),y])
    #print(real_prob)
    loss = (real_prob)/N + reg*(np.sum(W1_tmp*W1_tmp)+np.sum(W2_tmp*W2_tmp))
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
    grads = {}

    #self.params['W1'] = D, H
    #self.params['b1'] = H
    #self.params['W2'] = H,C
    #self.params['b2'] = C
    
    d_real_prob = 1/N
    d_scores = probs #(N,C)
    d_scores[range(N),y] -= 1
    d_scores = d_scores * d_real_prob #(N,C)
    dW2_1 = np.dot(X2_tmp.T, d_scores) #(H, N) (N,C) = H,C
    db2 = np.sum(d_scores,axis=0)
    dW2 = dW2_1 + 2*reg*W2_tmp
    
    dX2 = np.dot(d_scores, W2_tmp.T) #(N,C) (C,H)   =(N,H) 
    
    dX2_tmp = dX2
    dX2_tmp[X2_tmp <= 0] = 0 #(N,H)
    #print(X_tmp.T.shape)
    #print(dX2_tmp.shape)
    dW1 = np.dot(X_tmp.T, dX2_tmp) #(d,n) (n,h)
    
    db1 = np.sum(dX2_tmp,axis=0)
    dW1 += 2*reg*W1_tmp #(D,H)
    
    grads['W1'] = dW1
    grads['W2'] = dW2
    grads['b1'] = db1
    grads['b2'] = db2
    #grads = {dW1,db1,dW2,db2}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_iters: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []
    learning_rate_tmp = learning_rate
    for it in range(num_iters):
      X_batch = None
      y_batch = None
      randn = np.random.randint(num_train, size=batch_size)
      X_batch = X[randn,:]
      y_batch = y[randn]
      #loss, grads = self.loss(self,X_batch,y_batch,reg=reg)
     
      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      self.params['W1'] -= learning_rate_tmp*grads['W1']
      self.params['W2'] -= learning_rate_tmp*grads['W2']
      self.params['b1'] -= learning_rate_tmp*grads['b1']
      self.params['b2'] -= learning_rate_tmp*grads['b2']
      #learning_rate_tmp *= learning_rate_decay
      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      if verbose and it % 100 == 0:
        print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate_tmp *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None
    scores_y_pred = self.loss(X)
    y_pred = np.argmax(scores_y_pred, axis=1)
    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

    return y_pred

