# **2. XOR 클래스화**
## **2-1. XOR 클래스**

import numpy as np

class XOR():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.x2h = [] 
        self.x2h_bias = []
        self.h2y = []
        self.h2y_bias = []
        self.x2h_sig = []
        self.predict = []
        self.diff = []
        self.L = 0

    # Hidden Layer 설정
    def hidden_layer(self, k):
        # # X to Hidden Layer Weight
        self.x2h = np.random.rand(len(self.X), k)
        self.x2h_bias = np.random.rand(k, 1)
        # Hidden Layer to Y Weight
        self.h2y = np.random.rand(k, len(self.Y))
        self.h2y_bias = np.random.rand(1, len(self.Y))

        return self.x2h, self.x2h_bias, self.h2y, self.h2y_bias

    # Y 예측
    def predict_Y(self):
        # Sigmoid 함수
        def sigmoid(x):
	        return 1 / (1 + np.exp(-x))
         
        # X * X2H_Weight + X2H_Bias
        x2h_cal = np.dot(self.x2h.T, X) + self.x2h_bias
        self.x2h_sig = sigmoid(x2h_cal)
        # Sigmoid_X * H2Y_Weight + H2Y_Bias
        h2y_cal = np.dot(self.x2h_sig.T, self.h2y) + self.h2y_bias
        self.predict = sigmoid(h2y_cal)
        self.diff = self.predict - self.Y.T

        # Loss 함수
        def loss(r, p):
            return -((np.dot(r, np.log(p)) + np.dot((1 - r), np.log(1-p)))/len(r[0]))[0][0]
        
        self.L = loss(self.Y, self.predict)

        return self.predict, self.L

    # Back Propagation 구현
    def back_propagation(self, alpha):
        # Hidden Layer to Y Weight Gradient 
        self.h2y = self.h2y - alpha * np.dot(self.x2h_sig, self.diff)
        # Hidden layer to Y Bias Gradient
        self.h2y_bias = self.h2y_bias - alpha * self.diff
        # X to Hidden Layer Weight Gradient
        self.x2h = self.x2h - alpha * np.dot(X, np.dot(np.dot(np.dot(self.h2y, self.diff.T).T, self.x2h_sig), (1 - self.x2h_sig).T))
        # X to Hidden Layer Bias Gradient
        self.x2h_bias = self.x2h_bias - alpha * np.dot(np.dot(np.dot(self.h2y, self.diff.T).T, self.x2h_sig), (1-self.x2h_sig).T).T

    # 자동 실행
    def run(self, k, alpha, epochs):
        self.hidden_layer(k)
        L1 = 0
        count_epoch = 0
        for j in range(epochs):    
            self.predict_Y()
            if L1 != self.L:
                self.back_propagation(alpha)
                L1 = self.L.copy()
                count_epoch += 1
            else:
                break
        print("반복횟수 : {}".format(count_epoch))
        print("Loss : {}. 예측 Y값 : {}".format(self.L, self.predict.T))


if __name__== "__main__":
    docs = ['you will never know until you try']
    wv = Word2Vec(docs=docs, embedding_size=4)
