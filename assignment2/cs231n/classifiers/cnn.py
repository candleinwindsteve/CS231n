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
    - filter_size: Size of filters to use in the convolutional layer
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
    
    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################

    C, H, W = input_dim[0], input_dim[1], input_dim[2]

    # after the conv:
    pad = (filter_size - 1) / 2
    stride = 1
    H_f = (H + 2 * pad - filter_size) / stride + 1
    W_f = (W + 2 * pad - filter_size) / stride + 1
    #rint [H_f, W_f]

    # after the pooling:
    stride = 2
    pool_width,  pool_height = 2, 2
    H_f = (H_f - pool_height) / stride + 1
    W_f = (W_f - pool_width) / stride + 1

    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    self.params['W1'] = weight_scale * np.random.randn(num_filters, C, filter_size, filter_size)
    self.params['b1'] = np.zeros(num_filters)
    self.params['W2'] = weight_scale * np.random.randn(H_f * W_f * num_filters, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None

    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    cache = {}

    # conv - relu - 2x2 max pool - affine - relu - affine - softmax
    X, cache[0] = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
    X, cache[1] = affine_relu_forward(X, W2, b2)
    scores, cache[2] = affine_forward(X, W3, b3)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    N, reg = X.shape[0], self.reg

    loss, dout = softmax_loss(scores, y)
    for i in xrange(1, 4):
      loss += 0.5 * reg * np.sum(self.params['W'+str(i)]*self.params['W'+str(i)])

    dout, grads['W3'], grads['b3'] = affine_backward(dout, cache[2])
    dout, grads['W2'], grads['b2'] = affine_relu_backward(dout, cache[1])
    dout, grads['W1'], grads['b1'] = conv_relu_pool_backward(dout, cache[0])

    for i in xrange(1, 4):
      grads['W'+str(i)] += reg * self.params['W'+str(i)]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  

class ConvNet(object):

  """
  Implementation of :
    [conv-(bn)-relu-(dropout)-conv-(bn)-relu-(dropout)-pool]xN - [affine]xM - [softmax or SVM]
  """

  def __init__(self, input_dim=(3, 32, 32),
               num_filters=(16, 16, 32, 32, 64, 64, 128, 128),
               filter_sizes=(5, 5, 3, 3, 3, 3, 3, 3),
               hidden_dims=(100, 100, 100),
               num_classes=10,
               weight_scale=1e-3, reg=1e-3,
               use_batchnorm=True, use_dropout=True,
               dropout_p=0.5, dtype=np.float32,
               loss_func='softmax'):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """

    self.params = {}
    self.conv_params = {}
    self.bn_params = {}
    self.dropout_param = {}
    self.pool_params = {}
    self.reg = reg
    self.dtype = dtype
    self.use_batchnorm = use_batchnorm
    self.use_dropout = use_dropout
    self.loss_func = loss_func

    C, H, W = input_dim[0], input_dim[1], input_dim[2]

    self.num_conv_wrapper = len(filter_sizes)
    self.num_affine = len(hidden_dims)
    self.num_layers = self.num_conv_wrapper + self.num_affine + 1

    print 'Initializing ConvNetwork with:'
    print '[conv-(bn)-relu-(dropout)-conv-(bn)-relu-(dropout)-pool] X %d' % (self.num_conv_wrapper)
    print '[affine] X %d\n' % (self.num_affine)

    if self.use_batchnorm:
      print '***using spatial batch normalization'
    if self.use_dropout:
      print '***using dropout (p = %.2f)' % dropout_p
    if self.loss_func == 'softmax' or self.loss_func == 'svmloss':
      print '***using ''%s'' loss function' % loss_func
    else:
      raise RuntimeError("Loss function type not supported!")
    cnt = 0
    stride = 1
    H_out, W_out, C_out = H, W, C
    total_weight = 0
    total_memory = 0

    print '\nINPUT:'
    print 'memory: \t%d*%d*%d=%d' % (H, W, C, H*W*C)
    print 'weight: \t0\n'

    for i in xrange(self.num_conv_wrapper):
      num_filter = num_filters[i]
      filter_size = filter_sizes[i]
      self.params['W'+str(i+1)] = weight_scale * np.random.randn(num_filter, C_out, filter_size, filter_size)
      self.params['b'+str(i+1)] = np.zeros(num_filter)
      print '%d-th CONV Layer:' % (i+1)
      print 'memory: \t%d*%d*%d=%d' % (H_out, W_out, num_filter, H_out*W_out*num_filter)
      print 'weights: \t(%d*%d*%d)*%d=%d' % (filter_size, filter_size, C_out, num_filter,
                                              filter_size*filter_size*C_out*num_filter)
      total_memory += H_out*W_out*num_filter
      total_weight += filter_size*filter_size*C_out*num_filter
      conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
      self.conv_params.update({'conv_param'+str(i+1): conv_param})

      if self.use_batchnorm:
        bn_param = {'mode': 'train',
                    'running_mean': np.zeros(num_filter),
                    'running_var': np.zeros(num_filter),
                    }
        gamma = np.ones(num_filter)
        beta = np.zeros(num_filter)
        self.bn_params.update({'bn_param'+str(i+1): bn_param})
        self.params.update({
            'gamma'+str(i+1): gamma,
            'beta'+str(i+1): beta})

      if self.use_dropout:
        self.dropout_param = {'mode': 'train', 'p': dropout_p}

      if i % 2 == 1:  # time to apply pooling
        stride = 2
        pool_width, pool_height = 2, 2
        H_out = (H_out - pool_height) / stride + 1
        W_out = (W_out - pool_width) / stride + 1
        print '%d-th POOL Layer:' % (i / 2 + 1)
        print 'memory: \t%d*%d*%d=%d' % (H_out, W_out, num_filter,
                                      H_out*W_out*num_filter)
        print 'weights: \t0'
        total_memory += H_out*W_out*num_filter
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
        self.pool_params.update({'pool_param'+str(i+1): pool_param})

      C_out = num_filter

    D_out = H_out * W_out * num_filter

    cnt = self.num_conv_wrapper
    for i in xrange(self.num_affine + 1):
      if i == self.num_affine: # last FC layer to class scores
        self.params['W'+str(cnt+1)] = weight_scale * np.random.randn(D_out, num_classes)
        self.params['b'+str(cnt+1)] = np.zeros(num_classes)
        hidden_dim = num_classes # little trick
      else:
        hidden_dim = hidden_dims[i]
        self.params['W'+str(cnt+1)] = weight_scale * np.random.randn(D_out, hidden_dim)
        self.params['b'+str(cnt+1)] = np.zeros(hidden_dim)
      print '%d-th FC Layer:' % (i + 1)
      print 'memory: \t%d' % (hidden_dim)
      print 'weight: \t%d*%d=%d' % (D_out, hidden_dim, D_out*hidden_dim)
      total_memory += hidden_dim
      total_weight += D_out *hidden_dim

      cnt += 1
      D_out = hidden_dim

    print '\nSTATS:'
    if dtype == np.dtype('float64'):
      num_bytes = 8
    else:
      num_bytes = 4
    print 'Total memory: \t%d ~= %.2fMB (only forward! ~*2 for bwd)' % \
          (total_memory, 1.*total_memory*num_bytes/1024/1024)
    print 'Total params: \t%d ~= %.2fMB' % \
          (total_weight, 1.*total_weight*num_bytes/1024/1024)



    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    [conv-(bn)-relu-(dropout)-conv-(bn)-relu-(dropout)-pool]xN - [affine]xM - [softmax or SVM]

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """


    X = X.astype(self.dtype)
    mode = 'test' if y is None else 'train'

    # Set train/test mode for batchnorm params and dropout param since they
    # behave differently during training and testing.
    if self.use_dropout:
      if self.dropout_param is not None:
        self.dropout_param['mode'] = mode

    if self.use_batchnorm:
      for key, bn_param in self.bn_params.iteritems():
        bn_param['mode'] = mode


    scores = None

    cache = {}
    pool_cache = {}
    # we can have 2^2 = 4 cases here:
    for i in xrange(self.num_conv_wrapper):
      if self.use_batchnorm:
        if self.use_dropout:
          X, cache[i] = conv_batchnorm_relu_dropout_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                                                               self.conv_params['conv_param'+str(i+1)],
                                                               self.params['gamma'+str(i+1)],
                                                               self.params['beta'+str(i+1)],
                                                               self.bn_params['bn_param'+str(i+1)],
                                                               self.dropout_param)
        else:
          X, cache[i] = conv_batchnorm_relu_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                                                       self.conv_params['conv_param'+str(i+1)],
                                                       self.params['gamma'+str(i+1)],
                                                       self.params['beta'+str(i+1)],
                                                       self.bn_params['bn_param'+str(i+1)])
      else:
        if self.use_dropout:
          X, cache[i] = conv_relu_dropout_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                                                     self.conv_params['conv_param'+str(i+1)],
                                                     self.dropout_param)
        else:
          X, cache[i] = conv_relu_forward(X, self.params['W'+str(i+1)], self.params['b'+str(i+1)],
                                             self.conv_params['conv_param'+str(i+1)])

      if i % 2 == 1: # time to apply pooling
         X, pool_cache[i/2] = max_pool_forward_fast(X, self.pool_params['pool_param'+str(i+1)])

    cnt = self.num_conv_wrapper
    for i in xrange(self.num_affine + 1):
      if i == self.num_affine: # last FC layer to class scores
        scores, cache[cnt] = affine_forward(X, self.params['W'+str(cnt+1)], self.params['b'+str(cnt+1)])
      else:
        # print X.shape, self.params['W'+str(cnt+1)].shape, self.params['b'+str(cnt+1)].shape
        X, cache[cnt] = affine_forward(X, self.params['W'+str(cnt+1)], self.params['b'+str(cnt+1)])
      cnt += 1

    # print 'scores shape:'
    # print scores.shape

    if y is None:
      return scores

    loss, grads = 0, {}
    N, reg = X.shape[0], self.reg

    if self.loss_func == 'softmax':
      loss, dout = softmax_loss(scores, y)
    elif self.loss_func == 'svmloss':
      loss, dout = svm_loss(scores, y)
    else:
      raise RuntimeError("Loss function type not supported!")

    for i in xrange(1, self.num_layers + 1):
      loss += 0.5 * reg * np.sum(self.params['W'+str(i)]*self.params['W'+str(i)])

    # print 'loss = %f' % loss

    offset = self.num_conv_wrapper
    for i in reversed(xrange(self.num_affine + 1)):
      dout, grads['W'+str(offset+i+1)], grads['b'+str(offset+i+1)] = affine_backward(dout, cache[offset+i])

    for i in reversed(xrange(self.num_conv_wrapper)):
      if i % 2 == 1: # time to apply pooling backward
         dout = max_pool_backward_fast(dout, pool_cache[i/2])
        
      if self.use_batchnorm:
        if self.use_dropout:
          dout, grads['W'+str(i+1)], grads['b'+str(i+1)], \
          grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = conv_batchnorm_relu_dropout_backward(dout, cache[i])
        else:
          dout, grads['W'+str(i+1)], grads['b'+str(i+1)], \
          grads['gamma'+str(i+1)], grads['beta'+str(i+1)] = conv_batchnorm_relu_backward(dout, cache[i])
      else:
        if self.use_dropout:
          dout, grads['W'+str(i+1)], grads['b'+str(i+1)] = conv_relu_dropout_backward(dout, cache[i])
        else:
          dout, grads['W'+str(i+1)], grads['b'+str(i+1)] = conv_relu_backward(dout, cache[i])

    for i in xrange(1, self.num_layers + 1):
      grads['W'+str(i)] += reg * self.params['W'+str(i)]

    return loss, grads

