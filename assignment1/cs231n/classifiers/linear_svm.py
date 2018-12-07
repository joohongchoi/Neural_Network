import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  
  # X (N, D)
  # W (D, C)
  # Scores_mat (C, N)
  # y (N,)
  idx = np.linspace(0,num_classes-1,num_classes,dtype='int')
  #print(y)
  #print(np.arange(num_train))
  
  scores_mat = np.dot(X,W).T  
  
  #print(scores_mat[[y],np.linspace(0,num_train-1,num_train)])
  scores_mat = scores_mat - scores_mat[y,np.arange(num_train)] + 1 
  
  #scores_mat = scores_mat - y[None,:] + 1 
  scores_mat_relu = np.maximum(scores_mat, 0)
  #  scores_mat_cp = scores_mat.copy
  #scores_mat_cp[scores_mat_cp  < 0] = 0

  loss = np.sum(scores_mat_relu)/num_train

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  #loss /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  dW_1 = W*2*reg # D X C
  #row_v = np.sum(X, axis =0 ) # 1 X D
  #mat = np.dot( row_v[:,None], np.ones((1,num_classes)))
    # row_V = 1 X D
    # W = D X C
    # X = N X D
    # SCORE MAT = C X N

  one_mat = np.zeros(scores_mat_relu.shape)
  #print(one_mat.shape)
  
  one_mat[scores_mat_relu > 0] = 1/num_train
  #print(mat.shape)
  #print(one_mat.shape)
  dW_2 = np.dot(X.T , one_mat.T) # (D , C)= (c N)
  dW = dW_1 + dW_2
  #print(np.sum(W * W))

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  #print(X)
  num_train = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  scores_mat = np.dot(X,W).T  # C X N

  #idx = np.linspace(0,num_classes-1,num_classes,dtype=int16)
  #print(idx)
  scores_mat = scores_mat - scores_mat[y,np.arange(num_train)] + 1 
  

  scores_mat_relu = np.maximum(scores_mat, 0)
  #print(np.linspace(0,num_train-1,num_train, dtype =int))
  scores_mat_relu[y,np.arange(num_train)]=0
      
  loss = np.sum(scores_mat_relu)/num_train
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.

  loss += reg * np.sum(W * W)
  dW_1 = W*2*reg # D X C
  #row_v = np.sum(X, axis =0 ) # 1 X D
  #mat = np.dot( row_v[:,None], np.ones((1,num_classes)))
    # row_V = 1 X D
    # W = D X C
    # X = N X D
    # SCORE MAT = C X N

  one_mat = np.zeros(scores_mat_relu.shape)
  #print(one_mat.shape)
  
  one_mat[scores_mat_relu > 0] = 1/num_train
  #print(mat.shape)
  #print(one_mat.shape)
  dW_2 = np.dot(X.T , one_mat.T) # (D , C)= (c N)
  dW = dW_1 + dW_2

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
