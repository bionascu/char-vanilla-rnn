# Last changed: 05/25/2017
# Author: Beatrice Ionascu (bionascu@kth.se)

# DD2424 Assignment 4

import matplotlib.pyplot as plt
import numpy as np
import copy


class Parameters:

    def __init__(self, m, K):
        self.U = np.zeros((m, K))
        self.W = np.zeros((m, m))
        self.V = np.zeros((K, m))
        self.b = np.zeros((m, 1))
        self.c = np.zeros((K, 1))


class RNN (Parameters):

    def __init__(self, U, W, V, b, c, learning_rate, seq_length):
        Parameters.__init__(self, 0, 0)
        self.U = U
        self.W = W
        self.V = V
        self.b = b
        self.c = c
        self.learning_rate = learning_rate
        self.seq_length = seq_length
        self.m, self.K = U.shape
        self.mem = Parameters(self.m, self.K)

    def forward_pass(self, x, y, h0):
        """
        Compute the final and intermediary output vectors at each time step	and
        the loss for a single labelled sequence.
        """
        X, A, H, O, P = {}, {}, {}, {}, {}
        H[-1] = h0
        L = 0

        for t in xrange(self.seq_length):
            X[t] = np.zeros((self.K, 1))
            X[t][x[t]] = 1  # one hot encoding of input
            A[t] = np.dot(self.W, H[t - 1]) + np.dot(self.U, X[t]) + self.b  # [m x 1]
            H[t] = np.tanh(A[t])  # [m x 1]
            O[t] = np.dot(self.V, H[t]) + self.c  # [K x 1]
            P[t] = softmax(O[t])  # [K x 1]
            L += -np.log(P[t][y[t]])

        return X, A, H, P, L

    def backward_pass(self, X, y, A, H, P):

        grads = Parameters(self.m, self.K)
        grad_a = np.zeros((self.m, 1))  # [m x 1]

        for t in reversed(xrange(self.seq_length)):
            # Compute the gradient on the scores
            grad_scores = np.copy(P[t])  # [K x 1]
            grad_scores[y[t]] -= 1
            # Backpropagate the gradient to the parameters V and c
            grads.V += np.dot(grad_scores, H[t].T)
            grads.c += np.sum(grad_scores)
            # Recursively compute gradients for A[t] and H[t]
            if t == self.seq_length - 1:
                grad_h = np.dot(self.V.T, grad_scores)
            else:
                grad_h = np.dot(self.V.T, grad_scores) + np.dot(self.W.T, grad_a)
            grad_a = np.multiply(grad_h, 1 - np.multiply(H[t], H[t]))
            # Backpropagate the gradient to the parameters V, U, and b
            grads.W += np.dot(grad_a, H[t - 1].T)
            grads.U += np.dot(grad_a, X[t].T)
            grads.b += np.sum(grad_a)

        # Clip gradients to avoid exploding gradients
        for grad_param in [grads.U, grads.W, grads.V, grads.b, grads.c]:
            np.clip(grad_param, -5, 5, out=grad_param)

        return grads

    def gradient_check(self):
        raise NotImplementedError("Should have implemented this")

    def train(self, book_data, book_chars, syn_text_length, n_epochs):
        """
        Train the network using the vanilla version of mini-batch gradient
        descent and AdaGrad.
        """
        char_to_ind = {char: ind for ind, char in enumerate(book_chars)}
        ind_to_char = {ind: char for ind, char in enumerate(book_chars)}
        smooth_loss = -np.log(1.0 / len(book_chars)) * self.seq_length  # it 0
        smooth_loss_vector = []  # save smooth loss value for plotting

        it = 0  # iterations / update steps
        for ep in range(n_epochs):

            e = 0  # integer that keeps track of where in the book we are
            h0 = np.zeros((self.m, 1))  # (reset) initial hidden state

            while e + self.seq_length + 1 < len(book_data):

                # Define input and label sequences
                x = [char_to_ind[ch] for ch in book_data[e:e + self.seq_length + 1]]
                y = [char_to_ind[ch] for ch in book_data[e + 1:e + self.seq_length + 2]]

                # Sample from the model
                if it % 500 == 0:
                    x0 = np.zeros((self.K, 1))
                    x0[x[0]] = 1
                    ind_seq = synthesize_text(self, h0, x0, syn_text_length)
                    result = ''.join(ind_to_char[ind] for ind in ind_seq)
                    print '%s\n' % result

                # Forward and backward pass
                X, A, H, P, loss = self.forward_pass(x, y, h0)
                h0 = H[-1]
                smooth_loss = .999 * smooth_loss + .001 * loss
                smooth_loss_vector.append(smooth_loss)
                grad = self.backward_pass(X, y, A, H, P)

                # Print loss
                if it % 100 == 0:
                    print 'iter %d:\t loss: %f \n' % (it, smooth_loss)

                # Vanilla SGD update step using AdaGrad
                for param, grad_param, mem in zip([self.U, self.V, self.W, self.b, self.c],
                                                  [grad.U, grad.V, grad.W, grad.b, grad.c],
                                                  [self.mem.U, self.mem.V, self.mem.W, self.mem.b, self.mem.c]):
                    mem += grad_param * grad_param
                    param += -self.learning_rate * grad_param / np.sqrt(mem + 1e-8)

                e += self.seq_length
                it += 1

        return smooth_loss_vector


def sample(p):
    """
    Randomly sample a value from a probability distribution p
    """
    cp = np.cumsum(p)  # vector containing the cumul. sum of the probabilities
    a = np.random.rand()  # random draw representing the cp value of the sample
    ind = np.where(cp - a >= 0)[0][0]  # index of the sample
    return ind


def softmax(scores):
    """
    Interpret class scores as unnormalized log probabilities of the classes
    and compute the normalized class probabilities.
    """
    # scores -= np.max(scores) # avoid overflow by shifting scores to be < 0
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)  # [K x N]
    return probs


def synthesize_text(rnn, h0, x0, n):
    """
    Synthesize a sequence of n character indeces
    """
    h = h0  # vector the hidden state at time 0 [m x 1]
    x = x0  # vector representing the 1st (dummy) input vector to the RNN [Kx1]
    seq = []
    for t in xrange(n):
        a = np.dot(rnn.W, h) + np.dot(rnn.U, x) + rnn.b
        h = np.tanh(a)
        o = np.dot(rnn.V, h) + rnn.c
        p = softmax(o)
        ind = sample(p)
        # Append character index to sequence
        seq.append(ind)
        # Update input vector for the next time step
        x = np.zeros(x0.shape)
        x[ind] = 1
    return seq


def plot_loss(train_loss, n_epochs):
    epochs = np.arange(n_epochs + 1)
    plt.plot(epochs, train_loss, label="smooth loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc='upper right', shadow=True)
    plt.show()


def run():
    """
    Train a vanilla RNN network and plot results.
    """
    # Load data
    book_fname = 'data/goblet_book.txt'
    book_data = open(book_fname, 'r').read()
    book_chars = list(set(book_data))
    data_size, chars_size = len(book_data), len(book_chars)
    print 'The text has %d characters, %d unique.' % (data_size, chars_size)

    # Set hyperparameters
    m = 100  # dimensionality of the hidden state
    eta = 0.01  # learning rate (default 0.1)
    seq_length = 25  # length of input sequences used for training (default 25)

    # Set other gradient descent parameters
    n_epochs = 1

    # Initialize parameters as Gaussian random values (mean = 0 & s.d. = 0.01)
    K = chars_size
    rnd = np.random.RandomState(5)
    sig = 0.01
    U = rnd.normal(0, sig, (m, K))
    W = rnd.normal(0, sig, (m, m))
    V = rnd.normal(0, sig, (K, m))
    b = np.zeros((m, 1))
    c = np.zeros((K, 1))

    # Create a network instance
    rnn = RNN(copy.deepcopy(U), copy.deepcopy(W), copy.deepcopy(V),
              copy.deepcopy(b), copy.deepcopy(c), eta, seq_length)

    # # Gradient check
    # for layer_idx in range(L):
    # 	# Weights
    # 	print nn.gradient_check(train_data[:,:n_batch], train_labels[:n_batch],
    # 		"weight", layer_idx, batch_norm=True)
    # 	# Biases - when using batch norm only last layer matters
    # 	print nn.gradient_check(train_data[:,:n_batch], train_labels[:n_batch],
    # 		"bias", layer_idx, batch_norm=True)

    # Train network
    train_loss = rnn.train(book_data, book_chars, 200, n_epochs)

    # # Plot loss
    plot_loss(train_loss, n_epochs)


if __name__ == '__main__':
    # Train one network
    run()
