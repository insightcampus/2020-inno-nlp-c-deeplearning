import numpy as np

class XOR() :

    def _init_weights(self, X, h = 2):
        W = np.random.rand(X.shape[1], h)
        B = np.random.rand(1, h)
        return W, B

    def _affine (self, X, W, B):
        return np.dot(X, W) + B

    def _sigmoid (self, o):
        return 1/(1+np.exp(-o))

    def _eval_loss (self, Y, Y_hat):
        L = -(np.sum(Y.T*np.log(Y_hat) + (1-Y.T)*np.log(1-Y_hat)))/self.N
        return L

    def _gradients (self, X, Y, Y_hat, weights):
        W1, B1, W2, B2 = weights
        H1 = self.H1
        W2_g = np.dot(H1.T, Y_hat-Y.T)
        B2_g = np.mean(Y_hat-Y.T)
        W1_g = np.dot(X, (np.dot(W2, (Y_hat.T-Y))*H1.T*(np.ones((H1.shape[1], 1)) - H1.T)).T)
        B1_g = np.mean((np.dot(W2, (Y_hat.T-Y))*H1.T*(np.ones((H1.shape[1], 1)) - H1.T)).T, axis=0)
        return W1_g, B1_g, W2_g, B2_g

    def optimize (self, X, Y, h = 3, learning_rate = 0.1, epoch = 1000):
        # 초기 계산
        W1, B1 = self._init_weights(X.T, h)
        Z1 = self._affine(X.T, W1, B1)
        self.H1 = self._sigmoid(Z1)
        W2, B2 = self._init_weights(self.H1, 1)
        Z2 = self._affine(self.H1, W2, B2)
        Y_hat = self._sigmoid(Z2)
        self.N = Y_hat.shape[0]
        weights = W1, B1, W2, B2

        # 웨이트 업데이트
        for n in range(epoch):
            # gradient 계산
            gradients = self._gradients(X, Y, Y_hat, weights)

            # 업데이트
            for w, g in zip(weights, gradients):
                w -= learning_rate*g

            # 새롭게 Y_hat 예측
            Z1 = self._affine(X.T, W1, B1)
            self.H1 = self._sigmoid(Z1)
            Z2 = self._affine(self.H1, W2, B2)
            Y_hat = self._sigmoid(Z2)
            L = self._eval_loss(Y, Y_hat)

            # 100번에 한 번 loss 출력
            if n%100==0:
                print(np.sum(L))

        return weights, L, Y_hat

        
if __name__ == "__main__" :
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    Y = np.array([[0, 1, 1, 0]])
    xor = XOR()
    learned_weights, loss_trace, predicts = xor.optimize(X, Y, h=3, learning_rate = 0.1, epoch = 100000)
    print("Y hat : {}".format(predicts.T))
    print("predicts : {}".format([1 if y > 0.5 else 0 for y in predicts.T[0]]))