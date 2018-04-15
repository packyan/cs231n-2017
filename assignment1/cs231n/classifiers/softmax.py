import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  # compute the loss and the gradient
  num_classes = W.shape[1] #C
  num_train = X.shape[0] #N
  for i in range(num_train): # foreach train image
    scores = X[i].dot(W)
    scores -= np.max(scores) #because exp function results maybe very large.
    correct_class_score = scores[y[i]]
    loss = loss+ np.log(np.sum(np.exp(scores))) - correct_class_score
    dW[:, y[i]] -= X[i]
    s = np.exp(scores).sum()
    for j in range(num_classes): #f for each calss
      dW[:, j] += np.exp(scores[j]) / s * X[i] # X[i] (3073,) 1-dim

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  # Same with gradient
  dW /= num_train
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  N = X.shape[0]
  scores = X.dot(W) # (N,C)
  scores -= np.max(scores, axis = 1).reshape(N,1)
  s = np.exp(scores).sum(axis = 1)
  loss = np.sum(np.log(np.sum(np.exp(scores),axis = 1))) - np.sum(scores[range(N),y])
  print(scores.shape)
  print(scores[:,y].shape)
  print(scores[range(N),y].shape)
  #loss = np.log(s).sum() - scores[range(N), y].sum()
  counts = np.exp(scores) / s.reshape(N, 1)
  counts[range(N), y] -= 1
  dW = np.dot(X.T, counts)

  loss /= N
  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  dW = dW / N + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

