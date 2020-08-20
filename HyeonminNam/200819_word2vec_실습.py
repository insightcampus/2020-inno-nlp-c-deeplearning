import numpy as np
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import matplotlib as plt

class OneHotEncoder():
    def __init__(self, tokenized_docs):
        self.w2i = defaultdict(lambda : len(self.w2i))
        [self.w2i[w] for d in tokenized_docs for w in d]
        self.i2w = {v:k for k, v in self.w2i.items()}
        self.n_word = len(self.w2i)
    
    def encode(self, docs):
        ret = []
        for d in docs:
            for w in d:
                ret.append(self.get_onehot_vector(w))
        return ret

    def get_onehot_vector(self, w):
        v = [0] * len(self.w2i)
        v[self.w2i[w]] = 1
        return v

    def decode(self, v):
        pass


class Word2Vec():

    def __init__(self, docs, embedding_size, window_size, learning_rate = 0.1, epoch = 100):
        self.docs = docs
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.epoch = epoch

        self.tokenized_docs = [self._tokenizer(d) for d in docs]
        self.enc = OneHotEncoder(self.tokenized_docs)
        encoded_docs = self.enc.encode(self.tokenized_docs)
        W = self._init_weights(self.enc.n_word)
        X, Y = self._slide_window(encoded_docs)
        self.optimize(X, Y, W)

    def _tokenizer(self, sent):
        retokenize = RegexpTokenizer("[\w]+")
        token_lst = retokenize.tokenize(sent)
        return token_lst

    def _init_weights(self, n_word):
        W1 = np.random.rand(n_word, self.embedding_size)
        W2 = np.random.rand(self.embedding_size, n_word)
        return W1, W2

    def _slide_window(self, encoded_docs):
        context, center = [], []

        for i, w in enumerate(encoded_docs):
            s = max(0, i - self.window_size)
            e = min(len(encoded_docs), i + self.window_size)
            tmp_context = encoded_docs[s:e+1]
            tmp_center = tmp_context.pop(i-s)
            for c in tmp_context:
                center.append(tmp_center)
                context.append(c)

        return context, center

    def _input_to_hidden(self, X, W):
        return self._softmax(np.dot(X, W))

    def _hidden_to_output(self, H, W):
        return np.dot(H, W)

    def _eval_loss(self, Y, Y_hat):
        return -1/len(Y)*np.sum(Y*np.log(Y_hat))

    def _calc_gradients(self, X, Y, Y_hat, H, W2):
        err = Y_hat-Y
        dW2 = np.dot(H.T, err)
        dW1 = np.dot(np.dot(W2, err.T), X).T
        return dW1, dW2 

    def _softmax(self, O):
        return np.exp(O)/np.sum(np.exp(O), keepdims=True, axis=1)


    def optimize(self, X, Y, W):
        loss_trace = []
        for e in range(self.epoch):
            H = self._input_to_hidden(X, W[0])
            Y_hat = self._hidden_to_output(H, W[1])
            loss = self._eval_loss(Y, Y_hat)
            loss_trace.append(loss)
            gradients = self._calc_gradients(X, Y, Y_hat, H, W[1])

            for w, g in zip(W, gradients) :
                w += -self.learning_rate*g

        self.WV = W[0]

        plt.plot(loss_trace)
        plt.ylabel('loss')
        plt.xlabel('iterations (per hundreds)')
        plt.show

    def wv(self, w):
        return self.WV[self.enc.w2i[w]]
    
    def most_similar(self, w, n = 3) :
        v = self.wv(w)
        return np.dot(v, self.WV.T)

if __name__ == "__main__":
    docs = ['you will never know until you try']
    w2v = Word2Vec(docs, embedding_size = 4, window_size = 1)
    w2v.wv('you')
    w2v.most_similar('you')
