import numpy as np
import math
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


  num_train, dims = X.shape
  _, num_class = W.shape

  scores = X.dot(W)

  for i in xrange(num_train):
    f_i = scores[i, :]
    f_i -= np.max(f_i) # increase stability by shifting f values to have its maximum reach 0

    sum_exp_f_i_j = 0.0
    for j in xrange(num_class):
      sum_exp_f_i_j += math.exp(f_i[j])


    for j in xrange(num_class):
      p = np.exp(f_i[j]) / sum_exp_f_i_j
      dW[:, j] += X[i, :] * (p - (j == y[i]))

    loss += - math.log(math.exp(f_i[y[i]]) / sum_exp_f_i_j)

  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
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

  num_train, dims = X.shape
  _, num_class = W.shape

  scores = X.dot(W)
  scores -= np.max(scores, axis=1)[:, np.newaxis] # increase stability by shifting f values to have its maximum reach 0
  exps = np.exp(scores)
  row_sum_exps = np.sum(exps, axis=1)
  loss = np.sum(- np.log(np.exp(scores[range(num_train), y[range(num_train)]]) / row_sum_exps))

  probs = exps / row_sum_exps[:, np.newaxis]

  indicators = np.zeros_like(probs)
  indicators[range(num_train), y] = 1

  # dW: (D, C), X: (N, D), probs: (N, C),
  dW = X.T.dot(probs - indicators)

  """
    Add regularizations
  """
  loss /= num_train
  loss += reg * np.sum(W * W)

  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

