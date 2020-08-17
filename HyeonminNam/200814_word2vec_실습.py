import numpy as np
from nltk.tokenize import RegexpTokenizer
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class word2vec() :

    # 토큰화 함수
    def _tokenizer(self, sent):
        retokenize = RegexpTokenizer("[\w]+")
        token_lst = retokenize.tokenize(sent)
        return token_lst

    # onehot_vector 만들기
    def onehot_vector(self, token_lst):
        enc = OneHotEncoder()
        token_array = np.asarray(token_lst).reshape(-1, 1)
        enc.fit(token_array)
        onehot = enc.transform(token_array)
        onehot_array = onehot.toarray()
        return enc, onehot, onehot_array

    # X, Y 만들기
    def _XY(self, onehot_array, window):
        X = []
        Y = []
        for idx in range(len(onehot_array)):
            target = onehot_array[idx]
            for w in range(1, window+1):
                try:
                    X.append(onehot_array[idx-w])
                    Y.append(target)
                except:
                    pass
                try:
                    X.append(onehot_array[idx+w])
                    Y.append(target)
                except:
                    pass
        return np.asarray(X), np.asarray(Y)

    # 초기 weight 설정
    def _init_weights(self, onehot_array, n = 4):
        W1 = np.random.rand(onehot_array.shape[1], n)
        W2 = np.random.rand(n, onehot_array.shape[1])
        return W1, W2

    # softmax
    def _softmax(self, a):
        c = np.max(a)
        exp_a = np.exp(a-c)
        sum_exp_a = np.sum(exp_a)
        y = exp_a / sum_exp_a
        return y

    # loss 계산
    def _eval_loss (self, Y, Y_hat):
        L = -np.sum(self.Y*np.log(Y_hat))
        return L

    # gradient 계산
    def _gradients (self, Y_hat, weights):
        W1, W2 = weights
        W2_g = np.dot(self.H.T, (Y_hat-self.Y))
        W1_g = np.dot(self.X.T, np.dot((Y_hat-self.Y), W2.T))
        return W1_g, W2_g

    # 코사인 유사도 계산
    def cos_similarity(self, v1, v2):
        similarity = np.dot(v1, v2) / (np.sqrt(sum(np.square(v1))) * np.sqrt(sum(np.square(v2))))             
        return similarity

    # 최적화
    def optimize (self, sent, h = 4, window_size = 1, learning_rate = 0.1, epoch = 1000):
        # 토큰화
        token_lst = self._tokenizer(sent)
        
        # 초기값 설정
        enc, onehot, onehot_array = self.onehot_vector(token_lst)
        W1, W2 = self._init_weights(onehot_array, h)
        weights = W1, W2
        self.X, self.Y = self._XY(onehot_array, window_size)
        self.H = np.dot(self.X, W1)
        Z = np.dot(self.H, W2)
        Y_hat = np.apply_along_axis(self._softmax, 1, Z)

        # loss 저장용 리스트
        loss_lst = []

        # 웨이트 업데이트
        for n in range(epoch):

            # gradient 계산
            gradients = self._gradients(Y_hat, weights)

            # 업데이트
            for w, g in zip(weights, gradients):
                w -= learning_rate*g

            # 새롭게 Feedforward
            self.H = np.dot(self.X, W1)
            Z = np.dot(self.H, W2)
            # 각 array 별로 softmax 적용
            Y_hat = np.apply_along_axis(self._softmax, 1, Z)

            # 새로운 loss 계산
            L = self._eval_loss(self.Y, Y_hat)
            loss_lst.append(L)

            # 10번에 한 번 loss 출력
            if n%10==0:
                print(L)

        # 각 feature name과 임베딩 벡터 연결한 딕셔너리
        feature_name = [x[3:] for x in list(enc.get_feature_names())]
        w2v = {x:y for x, y in zip(feature_name, W1)}

        # 단어 임베딩 벡터 딕셔너리, loss 리스트 리턴
        return w2v, loss_lst



if __name__ == "__main__" :
    w2v = word2vec()
    sent = 'you will never know until you try'
    word2vec, loss_lst = w2v.optimize(sent, h=4, learning_rate = 0.1, epoch = 10000)

    print("Word2Vec : {}".format(word2vec))

    # Loss 그래프 출력
    x = [n for n in range(len(loss_lst))]
    plt.plot(x, loss_lst)
    plt.show()

    # 코사인 유사도 계산
    print('코사인 유사도:')
    print('you, will : {}'.format(w2v.cos_similarity(word2vec['you'], word2vec['will'])))
    print('never, try : {}'.format(w2v.cos_similarity(word2vec['never'], word2vec['try'])))
