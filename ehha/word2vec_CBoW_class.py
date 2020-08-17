import random
import numpy as np
import pandas as pd
from numpy import dot
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder


class Word2vec_CBoW():
    def __init__(self, docs='', h_node=4, learning_rate=0.01, epoch=100):
        self.docs = docs
        self.i_node = len(set(docs.split()))
        self.h_node = h_node
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss = 100

    def input_XY(self):
        input_list = np.asarray(self.docs.split()).reshape(-1, 1)

        # onehot encoding
        onehot = OneHotEncoder()
        onehot.fit(input_list)
        onehot.categories_
        onehot_vec = onehot.transform(input_list).toarray()

        # make X(input) data
        self.X_vec = []
        self.X_vec.append(onehot_vec[1])
        for i in range(len(onehot_vec) - 2):
            self.X_vec.append(onehot_vec[i])
            self.X_vec.append(onehot_vec[i+2])
        self.X_vec.append(onehot_vec[-2])

        X_vec_df = pd.DataFrame(self.X_vec)
        self.X = np.matrix(X_vec_df).T

        # make Y data
        Y_vec = []
        Y_vec.append(onehot_vec[0])
        for i in range(1, len(onehot_vec) - 1):
            Y_vec.append(onehot_vec[i])
            Y_vec.append(onehot_vec[i])
        Y_vec.append(onehot_vec[-1])

        Y_vec_df = pd.DataFrame(Y_vec)
        self.Y = np.matrix(Y_vec_df).T

    def __init_gradient(self):
        self.w1 = np.random.random(size=(self.i_node, self.h_node))
        self.w2 = np.random.random(size=(self.h_node, self.i_node))

    def cal_H(self):
        self.H = np.dot(self.w1.T, self.X)

    def cal_y_hat(self):
        WtH_cal = np.dot(self.w2.T, self.H)

        def softmax(wth):
            return np.exp(wth.T) / np.sum(np.exp(np.array(wth.T)), axis=1, keepdims=True)

        self.y_hat = softmax(WtH_cal)

    def cal_loss(self):
        self.loss = -np.multiply(self.Y, np.log(self.y_hat).T).sum() / len(self.X_vec)

    def cal_gradient(self):
        dif = self.y_hat.T - self.Y
        self.w1 = self.w1 - self.learning_rate * (self.X * (self.w2 * dif).T)
        self.w2 = self.w2 - self.learning_rate * (self.H * dif.T)

    def calc_similarity_matrix(self, vectors):
        def cosine_similarity(a, b):
            return dot(a, b.T) / (norm(a) * norm(b.T))

        n_word = len(vectors)
        similarity_matrix = np.zeros((n_word, n_word))

        for i in range(n_word):
            for j in range(i, n_word):
                similarity_matrix[j, i] = cosine_similarity(vectors[i], vectors[j]).round(4)
                # similarity_matrix[i, j] = similarity_matrix[j, i]
        return similarity_matrix

    def run(self):
        self.input_XY()
        self.__init_gradient()

        for i in range(self.epoch):
            before_loss = self.loss

            self.cal_H()
            self.cal_y_hat()
            self.cal_loss()
            self.cal_gradient()

            if self.loss == before_loss or self.loss == 0:
                print("loss : ", self.loss, "\n", "Y_hat : ", self.y_hat)
                break

        print("loss : ", self.loss, "\n", "Y_hat : ", self.y_hat)
        result_df = pd.DataFrame(self.calc_similarity_matrix(self.w1), columns=list(set(self.docs.split())), index=list(set(self.docs.split())))
        print(result_df)


docs = "you will never know until you try"

word2vec_cbow = Word2vec_CBoW(docs, learning_rate=0.01, epoch=100000)
word2vec_cbow.run()