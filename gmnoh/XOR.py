import numpy as np
import random

class XOR() :
    def _init_weights(self, h = 2):
        W1 = np.random.rand(2, h)
        B1 = np.random.rand(h, 1)
        W2 = np.random.rand(h, 1)
        B2 = np.random.rand(1, 1)

        return W1, W2, B1, B2

    def _affine (self, W, X, B):
        return W.T.dot(X) + B

    def _sigmoid (self, o):
        return 1. / (1 +np.exp(-o))

    def _eval_loss (self, Y, Y_hat):
        loss = 0
        for i in range(Y.shape[1]):
            loss += - ((Y[0][i]) * np.log(Y_hat[0][i]) + (1 - Y[0][i]) * np.log(1 - Y_hat[0][i]))

        return loss / Y.shape[1]

    def _gradients (self, X, Y, W1, W2, H, Y_hat):       
        # BackPropagate: Hidden Layer
        dW2 = np.dot(H, (Y_hat-Y).T)
        dB2 = 1. / Y.shape[1] * np.sum(Y_hat-Y, axis=1, keepdims=True)
        dH  = np.dot(W2, Y_hat-Y)
        # BackPropagate: Input Layer
        dZ1 = dH * H * (1-H)
        dW1 = np.dot(X, dZ1.T)
        dB1 = 1. / Y.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, dB1, dW2, dB2

    def optimize (self, X, Y, h = 3, learning_rate = 0.1, epoch = 100000):
        W1, W2, B1, B2 = self._init_weights(h)

        for i in range(epoch) : 
            Z1 = self._affine(W1, X, B1)
            H = self._sigmoid(Z1)
            Z2 = self._affine(W2, H, B2)
            Y_hat = self._sigmoid(Z2)

            loss = self._eval_loss(Y, Y_hat)
            dW1, dB1, dW2, dB2 = self._gradients(X, Y, W1, W2, H, Y_hat)
            W1, W2, B1, B2 = W1 - learning_rate * dW1, W2 - learning_rate * dW2, B1 - learning_rate * dB1, B2 - learning_rate * dB2,
            
            if i % 100 == 0 :
                print('Y hat: {}, loss: {}, predicts: {}'.format(Y_hat, loss, [1 if y > 0.5 else 0 for y in Y_hat[0]]))

if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    xor = XOR()
    # learned_weights, loss_trace, predicts = xor.optimize(X, Y, h=3, learning_rate = 0.1, epoch = 100000)
    xor.optimize(X, Y, h=3, learning_rate = 0.1, epoch = 100000)