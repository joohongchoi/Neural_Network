import numpy as np

class KNearestNeighbor(object):
  """ a kNN classifier with L2 distance """

  def __init__(self):
    pass

  def train(self, X, y):
    """
    Train the classifier. For k-nearest neighbors this is just 
    memorizing the training data.

    Inputs:
    - X: A numpy array of shape (num_train, D) containing the training data
      consisting of num_train samples each of dimension D.
    - y: A numpy array of shape (N,) containing the training labels, where
         y[i] is the label for X[i].
    """
    self.X_train = X
    self.y_train = y
    
  def predict(self, X, k=1, num_loops=2):
    """
    Predict labels for test data using this classifier.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data consisting
         of num_test samples each of dimension D.
    - k: The number of nearest neighbors that vote for the predicted labels.
    - num_loops: Determines which implementation to use to compute distances
      between training points and testing points.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    if num_loops == 0:
      dists = self.compute_distances_no_loops(X)
    elif num_loops == 1:
      dists = self.compute_distances_one_loop(X)
    elif num_loops == 2:
      dists = self.compute_distances_two_loops(X)
    else:
      raise ValueError('Invalid value %d for num_loops' % num_loops)

    return self.predict_labels(dists, k=k)

  def compute_distances_two_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a nested loop over both the training data and the 
    test data.

    Inputs:
    - X: A numpy array of shape (num_test, D) containing test data.

    Returns:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      is the Euclidean distance between the ith test point and the jth training
      point.
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]    
    dists = np.zeros((num_test, num_train))
    
    tmp_test = X
    tmp_train = self.X_train
    mult = np.dot(tmp_test,tmp_train.T)
    #dists = np.sqrt(np.square(tmp_test[:,np.newaxis]-tmp_train).sum(axis=2))
    
    tmp_test = np.sum((np.square(tmp_test)),axis=1)
    tmp_train = np.sum((np.square(tmp_train)),axis=1)
    
    #print(tmp_test.shape)
    #print(tmp_train.shape)
    
    #print(mult.shape)
    dists = np.sqrt(tmp_test.reshape(num_test,1) + tmp_train.reshape(1,num_train) - 2*mult)
    #ists = np.sqrt(tmp_test.reshape(1 + tmp_train - 2*mult)
    #print(dists.shape)
    #tmp_test(:,np.newaxis) - tmp_train    
    #tmp_test = X.T.reshape(-1,1).T
    #tmp_test = tmp_test.astype(uint16)
    #print(tmp_test.shape)
    #print(tmp_test)
    #tmp_test = np.tile(tmp_test,(num_train,1))
    #print(tmp_test.shape)
    
    #tmp_train = self.X_train.T.tile(num_test,1).T
    #tmp_train = tmp_train.tile((1,num_test)).T
    #dists = tmp_train-tmp_test 
    #dists = dists * dists
    #dists = dists.T.reshape((X.shape[1],-1))
    #dists = np.sum(dists, axis=0)
    #dists = dists.reshape((num_train,num_test))
    #for i in range(num_test):
     # for j in range(num_train):
        #####################################################################
        # TODO:                                                             #
        # Compute the l2 distance between the ith test point and the jth    #
        # training point, and store the result in dists[i, j]. You should   #
        # not use a loop over dimension.                                    #
        #####################################################################
        #pass
        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################

    return dists

  def compute_distances_one_loop(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using a single loop over the test data.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]    
    dists = np.zeros((num_test, num_train))
    
    tmp_test = X
    tmp_train = self.X_train
    mult = np.dot(tmp_test,tmp_train.T)
    #dists = np.sqrt(np.square(tmp_test[:,np.newaxis]-tmp_train).sum(axis=2))
    
    tmp_test = np.sum((np.square(tmp_test)),axis=1)
    tmp_train = np.sum((np.square(tmp_train)),axis=1)
    
    #print(tmp_test[:,np.newaxis].shape)
    #print(tmp_train.shape)
    
    #print(mult.shape)
    dists = np.sqrt(tmp_test[:,np.newaxis] + tmp_train - 2*mult)
    for i in range(num_test):
      #######################################################################
      # TODO:                                                               #
      # Compute the l2 distance between the ith test point and all training #
      # points, and store the result in dists[i, :].                        #
      #######################################################################
      pass
      #######################################################################
      #                         END OF YOUR CODE                            #
      #######################################################################
  
    return dists

  def compute_distances_no_loops(self, X):
    """
    Compute the distance between each test point in X and each training point
    in self.X_train using no explicit loops.

    Input / Output: Same as compute_distances_two_loops
    """
    num_test = X.shape[0]
    num_train = self.X_train.shape[0]    
    dists = np.zeros((num_test, num_train))
    
    tmp_test = X
    tmp_train = self.X_train
    mult = np.dot(tmp_test,tmp_train.T)
    #dists = np.sqrt(np.square(tmp_test[:,np.newaxis]-tmp_train).sum(axis=2))
    
    tmp_test = np.sum((np.square(tmp_test)),axis=1)
    tmp_train = np.sum((np.square(tmp_train)),axis=1)
    
    #print(tmp_test[:,np.newaxis].shape)
    #print(tmp_train.shape)
    
    #print(mult.shape)
    dists = np.sqrt(tmp_test[:,np.newaxis] + tmp_train - 2*mult)  
    #dists = np.dot(num_test,np.ones(1,num_train.shape[0])) - np.dot(num_train, np.ones(1,num_test.shape[0])
    #########################################################################
    # TODO:                                                                 #
    # Compute the l2 distance between all test points and all training      #
    # points without using any explicit loops, and store the result in      #
    # dists.                                                                #
    #                                                                       #
    # You should implement this function using only basic array operations; #
    # in particular you should not use functions from scipy.                #
    #                                                                       #
    # HINT: Try to formulate the l2 distance using matrix multiplication    #
    #       and two broadcast sums.                                         #
    #########################################################################
    pass
    #########################################################################
    #                         END OF YOUR CODE                              #
    #########################################################################
    return dists

  def predict_labels(self, dists, k=1):
    """
    Given a matrix of distances between test points and training points,
    predict a label for each test point.

    Inputs:
    - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
      gives the distance betwen the ith test point and the jth training point.

    Returns:
    - y: A numpy array of shape (num_test,) containing predicted labels for the
      test data, where y[i] is the predicted label for the test point X[i].  
    """
    num_test = dists.shape[0]
    y_pred = np.zeros(num_test)
    for i in range(num_test):
        # A list of length k storing the labels of the k nearest neighbors to
        # the ith test point.
        closest_y = np.empty(k)
        #print(i)
        #print(np.argmin(dists[i,:]))
        #print(self.y_train.shape)
        #print(self.y_train(np.argmin(dists[i,:])))
        tmp_arg = self.y_train[np.argsort(dists[i,:])]
        tmp = tmp_arg[0:k]
        #print(tmp)
        #print(np.bincount(np.array(tmp,  dtype=np.uint8)  ))
        bin_c = np.argmax(np.bincount(np.array(tmp,  dtype=np.uint8)  ))
        #print(closest_y)
        y_pred[i] = bin_c 
        #=closest_y[np.argmax(np.bincount(closest_y))]
        # self.y_train[np.argmin(dists[i,:])]
        
        #print(closest_y.shape)
        #y_pred[i] = closest_y
        pass
    pass
      #########################################################################
      # TODO:                                                                 #
      # Use the distance matrix to find the k nearest neighbors of the ith    #
      # testing point, and use self.y_train to find the labels of these       #
      # neighbors. Store these labels in closest_y.                           #
      # Hint: Look up the function numpy.argsort.                             #
      #########################################################################

      #########################################################################
      # TODO:                                                                 #
      # Now that you have found the labels of the k nearest neighbors, you    #
      # need to find the most common label in the list closest_y of labels.   #
      # Store this label in y_pred[i]. Break ties by choosing the smaller     #
      # label.                                                                #
      #########################################################################

    #########################################################################
      #                           END OF YOUR CODE                            # 
      #########################################################################

    return y_pred

