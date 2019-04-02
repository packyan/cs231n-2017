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
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        pass
        self.params['W1']= weight_scale*np.random.randn(input_dim,hidden_dim)
        self.params['b1']= np.zeros(hidden_dim)
        self.params['W2']= weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b2']= np.zeros(num_classes)
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
        
        W1, b1 = self.params['W1'], self.params['b1']
        
        W2, b2 = self.params['W2'], self.params['b2']
        
        layer1_out ,layer1_cache =  affine_relu_forward(X,W1,b1 )
        
        layer2_out ,layer2_cache =  affine_relu_forward(layer1_out,W2,b2)
        
        scores = layer2_out
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
        #loss part
        data_loss, dout = softmax_loss(layer2_out ,y)
        reg_loss = 0.5 * self.reg * (np.sum(W2 ** 2) + np.sum(W1 ** 2))
        loss = data_loss + reg_loss
        
        # back part
        dx2, dW2 , db2 = affine_relu_backward(dout, layer2_cache)
        
        dx1, dW1 , db1 = affine_relu_backward(dx2, layer1_cache)
        
        dW1 += self.reg * W1
        dW2 += self.reg * W2
        
        grads['W2'] = dW2
        grads['W1'] = dW1
        grads['b2'] = db2
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
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
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        #pass
        input_dim_copy = input_dim
        for layer in range(self.num_layers - 1):
        
            self.params['W%d' %(layer + 1)] = weight_scale * np.random.randn(input_dim_copy, hidden_dims[layer])
            self.params['b%d' %(layer + 1)] = np.zeros(hidden_dims[layer])
            
            #batch normalization intialize
            if(self.use_batchnorm ):
                self.params["gamma%d" %(layer+1)] = np.ones(hidden_dims[layer])
                self.params["beta%d" %(layer+1)] = np.zeros(hidden_dims[layer])  
                
            input_dim_copy = hidden_dims[layer]
            
        #initialization for  the final fc layer 
        self.params['W%d' %(self.num_layers)] = weight_scale * np.random.randn(input_dim_copy, num_classes)
        self.params['b%d' %(self.num_layers)] = np.zeros(num_classes)
        
                # Initialise the weights and biases for each fully connected layer connected to a Relu.
        # for i in range(self.num_layers - 1):
            # self.params['W' + str(i+1)] = np.random.normal(0, weight_scale, [input_dim, hidden_dims[i]])
            # self.params['b' + str(i+1)] = np.zeros([hidden_dims[i]])

            # if self.use_batchnorm:
                # self.params['beta' + str(i+1)] = np.zeros([hidden_dims[i]])
                # self.params['gamma' + str(i+1)] = np.ones([hidden_dims[i]])

            # input_dim = hidden_dims[i]  # Set the input dim of next layer to be output dim of current layer.

        # # Initialise the weights and biases for final FC layer
        # self.params['W' + str(self.num_layers)] = np.random.normal(0, weight_scale, [input_dim, num_classes])
        # self.params['b' + str(self.num_layers)] = np.zeros([num_classes])
        
        
        
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
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

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
        if self.use_batchnorm:
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
        # Flatten input images.
        input_data = X.reshape(X.shape[0],-1)
        
        affine_cache_list = []
        relu_cache_list = []
        bn_cache_list = []
        dropout_cache_list = []
        #print("forward pass:\n")
        # Do as many Affine-Relu forward passes as required (num_layers - 1).
        # Apply batch norm and dropout as required.
        for layer in range(self.num_layers - 1):
        
            Wi, bi = self.params['W%d'%(layer+1)], self.params['b%d'%(layer+1)]
            
            #affine layer
            affine_out, affine_cache = affine_forward(input_data, Wi, bi)
            affine_cache_list.append(affine_cache)
            
            #batch normlization:
            if(self.use_batchnorm):
                gammai, betai = self.params['gamma%d'%(layer+1)], self.params['beta%d'%(layer+1)]
                bn_out, bn_cache = batchnorm_forward(affine_out, gammai, betai, self.bn_param[layer])
                bn_cache_list.append(bn_cache)   
                #relu layer
                relu_out, relu_cache = relu_forward(bn_out)
                relu_cache_list.append(relu_cache)
            else:
              #relu layer
                relu_out, relu_cache = relu_forward(affine_out)
                relu_cache_list.append(relu_cache)
            #dropout
            if self.use_dropout:
                dropout_out, dropout_cache = dropout_forward(relu_out, self.dropout_param)
                dropout_cache_list.append(dropout_cache)
                input_data = dropout_out  
            else: 
                input_data = relu_out
           # print("layer: %d"%layer)
           # print(relu_cache_list[layer].shape) 
            
        Wi, bi = self.params['W%d'%(self.num_layers)], self.params['b%d'%(self.num_layers)]
        scores , affine_cache_final = affine_forward(input_data,Wi, bi )
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
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        pass
        
        data_loss, d_out = softmax_loss(scores,y)
        reg_loss = 0.0
        for i in range(self.num_layers):
            reg_loss += np.sum(self.params['W%d'%(i+1)]**2)
        loss = data_loss + 0.5*self.reg*reg_loss
        
        # Backprop dsoft to the last FC layer to calculate gradients.
        
        dx_last, dw_last, db_last = affine_backward(d_out, affine_cache_final)

        # Store gradients of the last FC layer
        grads['W'+str(self.num_layers)] = dw_last + self.reg*self.params['W'+str(self.num_layers)]
        grads['b'+str(self.num_layers)] = db_last
        
        dout = dx_last
        
        # use those to do backward
        #affine_cache_list = []
        #relu_cache_list = []
        #bn_cache_list = []
        #dropout_cache_list = []
        for layer in range(self.num_layers-1,0,-1):
            #drop
            if self.use_dropout:
                dout = dropout_backward(dout, dropout_cache_list[layer-1])
            #relu
            #print("dout.shape")
            #print(dout.shape)
            
            #print("relu layer: {}, shape {}".format(layer, relu_cache_list[1].shape))
            dout = relu_backward(dout,relu_cache_list[layer-1])
            
            if self.use_batchnorm:
                dout, dgamma, dbeta = batchnorm_backward(dout, bn_cache_list[layer-1])
                grads['gamma%d' % (layer)] = dgamma
                grads['beta%d' % (layer)] = dbeta

            #backforward
            dxi, dWi, dbi = affine_backward(dout, affine_cache_list[layer-1])
            dWi+=self.reg*self.params['W%d' % (layer)]

            grads['W%d' % (layer)] = dWi
            grads['b%d' % (layer)] = dbi

            dout = np.dot(dout, self.params['W%d' % (layer)].T)
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
