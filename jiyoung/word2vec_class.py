# import numpy as np
# from collections import defaultdict

# class OneHotEncoder():
#     def __init__(self, tokenized_docs):
#         self.w2i = defaultdict(lambda : len(self.w2i)) # 단어를 인덱스로 바꿈
#         [self.w2i[w] for d in tokenized_docs for w in d]
#         self.i2w = {v:k for k, v in self.w2i.items()} # 인덱스를 단어로 바꿈
#         self.n_word = len(self.w2i)
        
#     def encode(self, tokenized_docs):
#         ret = [] # 결과
#         for d in tokenized_docs:
#             for w in d:
#                 ret.append(self.get_onehot_vector(w))
#         return ret

#     def get_onehot_vector(self, w):
#         v = [0] * len(self.w2i)
#         v[self.w2i[w]] = 1
#         return v

#     def decode(self, v):
#         pass

# class Word2Vec():
#     def __init__(self, docs, embedding_size, window=1, learning_rate=0.1, epoch=100):
#         self.docs = docs
#         self.embedding_size = embedding_size
#         self.window = window
#         self.learning_rate = learning_rate
#         self.epoch = epoch
        
#         self.tokenized_docs = [d.split() for d in docs]
#         enc = OneHotEncoder(self.tokenized_docs)
#         encoded_docs = enc.encode(self.tokenized_docs)

#         W = self._init_weights(enc.n_word)
#         X, Y = self._slide_window(encoded_docs)
#         self._optimize(X, Y, W)

#     def _init_weights(self, n_word):
#         W1 = np.random.rand(n_word, self.embedding_size) # (6,4)
#         W2 = np.random.rand(self.embedding_size, n_word) # (4,6)
#         return W1, W2

#     def _slide_window(self, encoded_docs):
#         context, center = [], [] # skip, cbow는 반대

#         for i, w in enumerate(encoded_docs):
#             s = max(0, i - self.window) # 윈도우 시작위치 # 무조건 0이상
#             e = min(len(encoded_docs), i + self.window) # 윈도우 종료위치 # 무조건 문장길이를 넘지않는

#             tmp_context = encoded_docs[s:e+1] # 시작부터 종료까지
#             tmp_center = tmp_context.pop(i-s) # 중심단어를 뺌

#             for c in tmp_context:
#                 center.append(tmp_center)
#                 context.append(c) # c = tmp_context[i]

#         return context, center


#     def _input_to_hidden(self, X, W):
#         return np.dot(X, W)

#     def _hidden_to_output(self, H, W):
#         return self._softmax(np.dot(H, W))
        

#     def _eval_loss(self, Y, Y_hat):
#         return -1/len(Y) * np.sum(Y*np.log(Y_hat))
        

#     def _calc_gradients(self, X, Y, Y_hat, H, W2):
#         dW2 = 0
#         dW1 = 0

#         return dW1, dW2

#     def _softmax(self, O):
#         return np.exp(O)/np.sum(np.exp(O))

#     def _optimize(self, X, Y, W):
#         loss_trace = []

#         for e in range(self.epoch):
#             H = self._input_to_hidden(X, W[0])
#             Y_hat = self._hidden_to_output(H, W[1])
#             loss = self._eval_loss(Y, Y_hat)
#             loss_trace.append(loss)
#             gradients = self._calc_gradients(X, Y, Y_hat, H, W[1])
            
#             #update weights


# if __name__ == "__main__" :
#     docs = ["you will never know until you try"]
#     wv = Word2Vec(docs=docs, embedding_size=4)


import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

class OneHotEncoder() :
    def __init__(self, tokenized_docs) :
        self.w2i = defaultdict(lambda : len(self.w2i))
        [self.w2i[w] for d in tokenized_docs for w in d]
        self.i2w = {v:k for k, v in self.w2i.items()}
        self.n_word = len(self.w2i)

    def encode(self, tokenized_docs) :
        ret = []
        for d in tokenized_docs :
            for w in d :
                ret.append(self.get_onehot_vector(w))
        return ret

    def get_onehot_vector(self, w) :
        v = [0] * len(self.w2i)
        v[self.w2i[w]] = 1
        return v

    def decode(self, v) :
        pass


class Word2Vec():
    def __init__(self, docs, embedding_size, window=1, learning_rate = 0.1, epoch=100) :
        self.embedding_size = embedding_size
        self.window = window
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.tokenized_docs = [d.split() for d in docs]
        self.enc = OneHotEncoder(self.tokenized_docs)
        encoded_docs = self.enc.encode(self.tokenized_docs)
        W = self._init_weights(self.enc.n_word)
        X, Y = self._slide_window(encoded_docs)
        self._optimize(X, Y, W)

    def _init_weights(self, n_word):
        W1 = np.random.rand(n_word, self.embedding_size)
        W2 = np.random.rand(self.embedding_size, n_word)
        return W1, W2

    def _slide_window(self, encoded_docs):
        context, center = [], []
        for i, w in enumerate(encoded_docs) :
            s = max(0, i - self.window)
            e = min(len(encoded_docs), i + self.window)
            tmp_context = encoded_docs[s:e+1]
            tmp_center = tmp_context.pop(i-s)
            for c in tmp_context :
                center.append(tmp_center)
                context.append(c)
        return np.array(context), np.array(center)

    def _input_to_hidden(self, X, W):
        return np.dot(X, W)

    def _hidden_to_output(self, H, W):
        return self._softmax(np.dot(H, W))

    def _eval_loss(self, Y, Y_hat) :
        return -1/len(Y) * np.sum(Y*np.log(Y_hat))

    def _calc_gradients(self, X, Y, Y_hat, H, W2) :
        err = Y_hat - Y

        dW2 = np.dot(H.T, err)
        dW1 = np.dot(np.dot(W2, err.T), X).T


        return dW1, dW2

    def _softmax(self, O) :
        return np.exp(O)/np.sum(np.exp(O), keepdims=True, axis=1)

    def _optimize(self, X, Y, W) :
        loss_trace = []

        for e in range(self.epoch) :
            H = self._input_to_hidden(X, W[0])
            Y_hat = self._hidden_to_output(H, W[1])
            loss = self._eval_loss(Y, Y_hat)
            loss_trace.append(loss)
            gradients = self._calc_gradients(X, Y, Y_hat, H, W[1])

            for w, g in zip(W, gradients):
                w += -self.learning_rate * g

        self.WV = W[0]

        plt.plot(loss_trace)
        plt.ylabel('loss')
        plt.xlabel('iterations (per hundreds)')
        plt.show()

    def wv(self, w):
        return self.WV[self.enc.w2i[w]]

    def most_similar(self, w, n=3):
        v = self.wv(w)


        return np.dot(v, self.WV.T)

            #update weights



if __name__ == "__main__" :
    docs = ["you will never know until you try"]
    w2v = Word2Vec(docs=docs, embedding_size=4)

    w2v.wv('you')
    w2v.most_similar('you')
















