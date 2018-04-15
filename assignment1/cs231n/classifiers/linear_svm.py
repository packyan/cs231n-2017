import numpy as np
from random import shuffle
from past.builtins import xrange

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
  for i in xrange(num_train): # foreach train image
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes): #f for each calss
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1 margin:L_ij
      if margin > 0:
        loss += margin
        dW[:, j] += X[i] # X[i] (3073,) 1-dim
        dW[:, y[i]] -= X[i]
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Same with gradient
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
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

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  # num_train = X.shape[0]
  # delta = 1.0
  # for i in range(num_train)
	 #  scores = W.dot(X[i])
	 #  margins = np.maximum(0, scores - scores[y] + delta)
	 #  margins[y] = 0
  # 	  loss += np.sum(margins)
  delta = 1.0
  N = X.shape[0]
  scores = X.dot(W) # (N,C)
  margin = scores - scores[range(0,N), y].reshape(-1, 1) + 1 # N x C
  margin[range(N), y] = 0
  margin = (margin > 0) * margin
  loss += margin.sum()/N
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # Add regularization to the loss.
  loss += 0.5*reg * np.sum(W * W)
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


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
  counts = (margin > 0).astype(int)
  counts[range(N), y] = - np.sum(counts, axis = 1)
  dW += np.dot(X.T, counts)
  # Same with gradient
  dW /= N
  dW += reg * W 
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
