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
    num_classes = W.shape[1]

    for i in range(len(X)):
        prediction = X[i].dot(W)
        prediction -= np.max(prediction)
        softmax = np.exp(prediction[y[i]]) / np.exp(prediction).sum()
        loss -= np.log(softmax)

        probabilities = np.exp(prediction) / np.exp(prediction).sum()
        for c in range(num_classes):
            delta = probabilities[c]
            if c == y[i]:
                delta -= 1
            else:
                dW[:, c] += delta * X[i]
        # for c in range(num_classes):
        #     if c == y[i]:
        #         delta = probabilities[c] - 1                
        # else:
        #     delta = probabilities[c]
        #     dW[:, c] += delta * X[i]

    loss = (loss / len(X)) + reg * np.linalg.norm(W)
    dW = (dW / len(X)) + reg * W
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
    eps = 1e-14
    N = len(X)

    predictions = X.dot(W)
    predictions -= predictions.max(axis=1, keepdims=True)
    probabilities = np.exp(predictions) / np.exp(predictions).sum(axis=1, keepdims=True)

    loss = -np.log(probabilities[np.arange(N), y] + eps).sum() / N

    deltas = probabilities.copy()
    deltas[np.arange(N), y] -= 1
    deltas /= N
    dW = X.T.dot(deltas)
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return loss, dW

