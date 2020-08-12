

import numpy as np
class XOR() :
    def _init_weights(self, h = 2):
        W1 = np.random.rand(2, h)
        B1 = np.random.rand(h, 1)
        W2 = np.random.rand(h, 1)
        B2 = np.random.rand(1, 1)

        # W1 = np.array([[0.5, 0], [0.5, 0]])
        # B1 = np.array([[0.6], [0.6]])
        # W2 = np.array([[0.7], [0.7]])
        # B2 = np.array([[0.8]])

        return W1, B1, W2, B2

    def _affine(self, W, X, B):
      return np.dot(W.T, X) + B

    def _sigmoid(self, o):
      return 1./(1 + np.exp(-1*o))

    def _eval_loss(self, X, Y, weights):
        W1, B1, W2, B2 = weights
        # forward propagation : input layer
        Z1 = self._affine(W1, X, B1)
        H = self._sigmoid(Z1)

        # forward propagation : hidden layer
        Z2 = self._affine(W2, H, B2)
        Y_hat = self._sigmoid(Z2)

        loss = (-1) * 1./X.shape[1] * np.sum(Y*np.log(Y_hat) + (1-Y)*np.log(1-Y_hat))
        return Z1, H, Z2, Y_hat, loss
    
    def _gradients(self, X, Y, weights, Z1, H, Z2, Y_hat):
      W1, B1, W2, B2 = weights
    
      # BackPropagate: Hidden Layer
      dW2 = np.dot(H, (Y_hat-Y).T)
      dB2 = 1. / Y.shape[1] * np.sum(Y_hat-Y, axis=1, keepdims=True)
      dH  = np.dot(W2, Y_hat-Y)
      # BackPropagate: Input Layer
      dZ1 = dH * H * (1-H)
      dW1 = np.dot(X, dZ1.T)
      dB1 = 1. / Y.shape[1] * np.sum(dZ1, axis=1, keepdims=True)
      return [dW1, dB1, dW2, dB2]


if __name__ == "__main__" :
    X = np.array([[0,0,1,1], [0,1,0,1]])
    Y = np.array([[0,1,1,0]])

    xor = XOR()
    W1, B1, W2, B2 = xor._init_weights(2)

    epoch = 100000
    learning_rate = 0.1

    for i in range(epoch) :
        if i == 1 :
            print("test")
        #weights = (W1, B1, W2, B2)
        Z1, H, Z2, Y_hat, loss = xor._eval_loss(X, Y, (W1, B1, W2, B2))

        #print(W1, B1, W2, B2)

        gradient = xor._gradients(X, Y, (W1, B1, W2, B2), Z1, H, Z2, Y_hat)

        W1 = W1 - learning_rate * gradient[0]
        B1 = B1 - learning_rate * gradient[1]
        W2 = W2 - learning_rate * gradient[2]
        B2 = B2 - learning_rate * gradient[3]


        if i % 1000 == 0 : 
            print(Y_hat)
            print(W1)
            print(loss)

    Z1, H, Z2, Y_hat, loss = xor._eval_loss(X, Y, (W1, B1, W2, B2))
    print(Y_hat)



