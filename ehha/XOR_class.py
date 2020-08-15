import numpy as np
import random


class XOR():
    def __init__(self, X, Y, i_node=2, h_node=2, learning_rate=0.1, epoch=100):
        self.X = X
        self.Y = Y
        self.i_node = i_node
        self.h_node = h_node
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.loss = 100
        self.n = len(Y[0])

    def __init_gradient(self):
        self.w1 = np.random.random(size=(self.i_node, self.h_node))
        self.w2 = np.random.random(size=(self.h_node, 1))
        self.b1 = np.random.random(size=(self.h_node, 1))
        self.b2 = np.random.random(size=(1, 1))

    def cal_h(self):
        sum_h = np.dot(self.w1.T, X) + self.b1
        self.h = 1 / (1 + np.exp(-sum_h))

    def cal_y_hat(self):
        sum_y_hat = np.dot(self.w2.T, self.h) + self.b2
        self.y_hat = 1 / (1 + np.exp(-sum_y_hat))

    def cal_loss(self):
        self.loss = -1 / self.n * sum(np.dot(Y, np.log(self.y_hat).T) + np.dot((1 - Y), np.log(1 - self.y_hat).T))

    def cal_gradient(self):
        # Gradient 계산
        dif = self.y_hat - Y
        c_w1 = np.dot(X, (np.dot(self.w2, dif) * self.h * (1 - self.h)).T)
        c_w2 = np.dot(self.h, dif.T)
        c_b1 = np.dot(self.w2, dif) * self.h * (1-self.h)

        # update weight
        self.w1 = self.w1 - self.learning_rate * c_w1
        self.w2 = self.w2 - self.learning_rate*c_w2
        self.b1 = self.b1 - self.learning_rate * c_b1
        self.b2 = self.b2 - self.learning_rate * dif

    def run(self):
        self.__init_gradient()

        for i in range(self.epoch):
            before_loss = self.loss

            self.cal_h()
            self.cal_y_hat()
            self.cal_loss()
            self.cal_gradient()

            if self.loss == before_loss or self.loss == 0:
                print("loss : ", self.loss, "\n", "Y_hat : ", self.y_hat)
                break

        print("loss : ", self.loss, "\n", "Y_hat : ", self.y_hat)


X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

xor = XOR(X, Y, learning_rate=0.1, epoch=100000)
xor.run()


