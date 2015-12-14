from numpy import *
import numpy as np
from nn.base import NNBase
from nn.math import softmax, make_onehot
from misc import random_weight_matrix


##
# Evaluation code; do not change this
##
from sklearn import metrics
def full_report(y_true, y_pred, tagnames):
    cr = metrics.classification_report(y_true, y_pred,
                                       target_names=tagnames)
    print cr

def eval_performance(y_true, y_pred, tagnames):
    pre, rec, f1, support = metrics.precision_recall_fscore_support(y_true, y_pred)
    print "=== Performance (omitting 'O' class) ==="
    print "Mean precision:  %.02f%%" % (100*sum(pre[1:] * support[1:])/sum(support[1:]))
    print "Mean recall:     %.02f%%" % (100*sum(rec[1:] * support[1:])/sum(support[1:]))
    print "Mean F1:         %.02f%%" % (100*sum(f1[1:] * support[1:])/sum(support[1:]))


##
# Implement this!
##
class WindowMLP(NNBase):
    """Single hidden layer, plus representation learning."""

    def __init__(self, wv, windowsize=3,
                 dims=[None, 100, 5],
                 reg=0.001, alpha=0.01, rseed=10):
        """
        Initialize classifier model.

        Arguments:
        wv : initial word vectors (array |V| x n)
            note that this is the transpose of the n x |V| matrix L
            described in the handout; you'll want to keep it in
            this |V| x n form for efficiency reasons, since numpy
            stores matrix rows continguously.
        windowsize : int, size of context window
        dims : dimensions of [input, hidden, output]
            input dimension can be computed from wv.shape
        reg : regularization strength (lambda)
        alpha : default learning rate
        rseed : random initialization seed
        """

        # Set regularization
        self.lreg = float(reg)
        self.alpha = alpha # default training rate

        dims[0] = windowsize * wv.shape[1] # input dimension
        param_dims = dict(W=(dims[1], dims[0]),
                          b1=(dims[1],),
                          U=(dims[2], dims[1]),
                          b2=(dims[2],),
                          )
        param_dims_sparse = dict(L=wv.shape)

        # initialize parameters: don't change this line
        NNBase.__init__(self, param_dims, param_dims_sparse)

        random.seed(rseed) # be sure to seed this for repeatability!

        # any other initialization you need

    def _acc_grads(self, window, label):
        """
        Accumulate gradients, given a training point
        (window, label) of the format

        window = [x_{i-1} x_{i} x_{i+1}] # three ints
        label = {0,1,2,3,4} # single int, gives class

        Your code should update self.grads and self.sgrads,
        in order for gradient_check and training to work.

        So, for example:
        self.grads.U += (your gradient dJ/dU)
        self.sgrads.L[i] = (gradient dJ/dL[i]) # this adds an update for that index
        
        """
        W, b1, U, b2 = self.params.W, self.params.b1, self.params.U, self.params.b2
        H, window_size = W.shape
        C, H = U.shape
        X = self.sparams.L[window].reshape(window_size, 1)

        # Forward Pass (predictions)
        z = np.dot(W, X) + b1.reshape(H, 1)
        #hidden = 1. / (1.+np.exp(-z))
        hidden = np.tanh(z)
        
        # Tanh
        scores = np.dot(U, hidden) + b2.reshape(C, 1)
        probs = softmax(scores)
        y_hat = probs[label]

        # Cross Entropy Loss
        loss = -np.log(y_hat)

        # Backpropagate!
        dscores = probs
        dscores[label] -= 1
        
        self.grads.b2 += dscores.reshape(C)
        self.grads.U += np.dot(dscores, hidden.T)
        
        dhidden = np.dot(U.T, dscores)
        #dz = (hidden * (1-hidden)) * dhidden # sigmoid
        dz = (1 - hidden**2) * dhidden # tanh
        
        self.grads.b1 += dz.reshape(H)
        self.grads.W += np.dot(dz, X.T)
        
        # Regularization
        loss += 0.5 * self.lreg*(np.sum(W**2) + np.sum(b1**2) + np.sum(U**2) + np.sum(b2**2))
        
        self.grads.W  += self.lreg*W
        self.grads.b1 += self.lreg*b1
        self.grads.U  += self.lreg*U
        self.grads.b2 += self.lreg*b2

    def predict_proba(self, windows):
        """
        Predict class probabilities.

        Should return a matrix P of probabilities,
        with each row corresponding to a row of X.

        windows = array (n x windowsize),
            each row is a window of indices
        """
        # handle singleton input by making sure we have
        # a list-of-lists
        if not hasattr(windows[0], "__iter__"):
            windows = [windows]

        #### YOUR CODE HERE ####

        print windows

        #### END YOUR CODE ####

        return P # rows are output for each input


    def predict(self, windows):
        """
        Predict most likely class.
        Returns a list of predicted class indices;
        input is same as to predict_proba
        """

        #### YOUR CODE HERE ####


        #### END YOUR CODE ####
        return c # list of predicted classes


    def compute_loss(self, windows, labels):
        """
        Compute the loss for a given dataset.
        windows = same as for predict_proba
        labels = list of class labels, for each row of windows
        """

        #### YOUR CODE HERE ####

        W, b1, U, b2 = self.params.W, self.params.b1, self.params.U, self.params.b2
        H, window_size = W.shape
        C, H = U.shape
        X = self.sparams.L[windows].reshape(window_size, 1)

        # Forward Pass (predictions)
        z = np.dot(W, X) + b1.reshape(H, 1)
        #hidden = 1. / (1.+np.exp(-z)) # sigmoid
        hidden = np.tanh(z) # sigmoid
        
        # Tanh
        scores = np.dot(U, hidden) + b2.reshape(C, 1)
        probs = softmax(scores)
        y_hat = probs[labels]
        
        return -np.log(y_hat)