from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from numpy import array
import numpy as np
import random


class word2vec():
    def __init__(self, doc, h = 4, win_size = 1, learning_rate = 0.1, epoch = 10000):
        self.doc = doc
        self.h = h
        self.win_size = win_size
        self.alpha = learning_rate
        self.epoch = epoch

    def _one_hot_encoding(self):
        words = array(self.doc.split())
        
        label_enc = LabelEncoder()
        int_enc = label_enc.fit_transform(words)

        onehot = OneHotEncoder(sparse = False)
        int_enc = int_enc.reshape(len(int_enc), 1)
        onehot = onehot.fit_transform(int_enc)

        return onehot.T

    def _init_weights(self, onehot):
        W1 = np.random.rand(onehot.shape[0], self.h)
        W2 = np.random.rand(self.h, onehot.shape[0])

        return W1, W2

    def _affine(self, X, W):
        return W.T.dot(X)

    def _softmax(self, O):
        return (np.exp(O.T) / np.sum(np.exp(O.T), keepdims = True, axis = 1)).T

    def _loss(self, Y, Y_hat):
        loss = 0
        for i in range(len(Y_hat[0])):
            loss -= Y.T.dot(np.log(Y_hat.T[i]))
        
        return loss

    def _gradients(self, X, Y, Y_hat, W2, H):
        dW1 = np.dot(X, (W2.dot(Y_hat - Y)).T)
        dW2 = np.dot(H, (Y_hat - Y).T)

        return dW1, dW2

    def optimize(self):
        onehot = self._one_hot_encoding()
        W1, W2 = self._init_weights(onehot)

        self.loss_trace = []

        for e in range(self.epoch):
            for i in range(len(onehot)):
                X = np.concatenate((onehot[:, (i-self.win_size):i], onehot[:, (i+1):(i+self.win_size+1)]), axis = 1)
                Y = onehot[:, i:i+1]

                H = self._affine(X, W1)
                Z = self._affine(H, W2)
                Y_hat = self._softmax(Z)
                loss = self._loss(Y, Y_hat)

            self.loss_trace.append(loss / len(onehot))

            dW1, dW2 = self._gradients(X, Y, Y_hat, W2, H)
            W1, W2 = W1 - self.alpha * dW1, W2 - self.alpha * dW2

    def plot_loss(self):
        plt.plot([i for i in range(self.epoch)], self.loss_trace)
        plt.title('Loss trace')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()


if __name__ == "__main__" :
    doc = 'you will never know until you try'

    w2v = word2vec(doc)
    w2v.optimize()
    w2v.plot_loss()
