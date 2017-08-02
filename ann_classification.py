import numpy as np


class Config:
    nn_input_dim = 2  # input layer dimensionality
    nn_output_dim = 2  # output layer dimensionality
    # Gradient descent parameters (I picked these by hand)
    epsilon = 0.01  # learning rate for gradient descent
    reg_lambda = 0.01  # regularization strength

class Model:
    def __init__(self, nn_hdim):
        '''
            Args:
                nn_hdim: Number of nodes in the hidden layer
        '''
        W1 = np.random.randn(Config.nn_input_dim, nn_hdim) / np.sqrt(Config.nn_input_dim)
        b1 = np.zeros((1, nn_hdim))
        W2 = np.random.randn(nn_hdim, Config.nn_output_dim) / np.sqrt(nn_hdim)
        b2 = np.zeros((1, Config.nn_output_dim))

        self.weights = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

    def calculate_loss(self, X, y):
        'Helper function to evaluate the total loss on the dataset'

        weights = self.weights
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

        num_examples = len(X)  # training set size

        # Forward propagation to calculate our predictions
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        # Calculating the loss
        corect_logprobs = -np.log(probs[range(num_examples), y])
        data_loss = np.sum(corect_logprobs)
        # Add regulatization term to loss (optional)
        data_loss += Config.reg_lambda / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
        return 1. / num_examples * data_loss

    def fit(self, X, y, num_passes=20000, print_loss=False):
        '''
            This function learns parameters for the neural network and update the weights of the model.
                num_passes: Number of passes through the training data for gradient descent
                print_loss: If True, print the loss every 1000 iterations
        '''
        
        # Initialize the parameters to random values. We need to learn these.
        num_examples = len(X)
        np.random.seed(0)

        weights = self.weights
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

        # Gradient descent. For each batch...
        for i in range(0, num_passes):
            # Forward propagation
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = probs
            delta3[range(num_examples), y] -= 1
            dW2 = a1.T @ delta3
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3 @ W2.T * (1 - np.power(a1, 2))
            dW1 = X.T @ delta2
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += Config.reg_lambda * W2
            dW1 += Config.reg_lambda * W1

            # Gradient descent parameter update
            W1 += -Config.epsilon * dW1
            b1 += -Config.epsilon * db1
            W2 += -Config.epsilon * dW2
            b2 += -Config.epsilon * db2

            # Assign new parameters to the model
            weights = {
                'W1': W1,
                'b1': b1,
                'W2': W2,
                'b2': b2
            }

            # Optionally print the loss.
            # This is expensive because it uses the whole dataset, so we don't want to do it too often.
            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))

    def predict(self, x):
        weights = self.weights
        W1, b1, W2, b2 = weights['W1'], weights['b1'], weights['W2'], weights['b2']

        # Forward propagation
        z1 = x @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)
