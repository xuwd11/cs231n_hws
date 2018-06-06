import numpy as np
from random import shuffle

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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  y_dummy = np.zeros((num_train, num_classes))
  y_dummy[np.arange(num_train), y] = 1
  for i in range(num_train):
    p = X[i].dot(W)
    p -= np.max(p)
    p = np.exp(p)
    p /= np.sum(p)
    loss -= np.log(p[y[i]])
    dW += X[i].reshape(-1, 1) * (p - y_dummy[i]).reshape(1, -1)
  loss /= num_train
  loss += reg * np.sum(np.square(W))
  dW /= num_train
  dW += 2 * reg * W
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
  num_train = X.shape[0]
  num_classes = W.shape[1]
  y_dummy = np.zeros((num_train, num_classes))
  y_dummy[np.arange(num_train), y] = 1

  p = X.dot(W)
  p -= np.max(p)
  p = np.exp(p)
  p /= p.sum(axis=1, keepdims=True)
  loss = - np.mean(np.log(p[np.arange(num_train), y])) + reg * np.sum(np.square(W))

  dW = X.T.dot(p - y_dummy) / num_train + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

