import numpy as np
import random
import matplotlib.pyplot as plt

class Word2Vec_CBow() : 
    
    def __init__(self, H = 4, learning_rate = 0.1) :
        self.H = H
        self.learning_rate = learning_rate
        self.loss_ls = [] #loss 변화량 그래프로 보여주기위한 리스트
        return
    
    def tokenize(self, doc) : # 토크나이즈 수정
        tokens = doc.split()
        return tokens

    def word_indexing(self,tokens) :
        word2index = {}
        for voca in tokens :
            if voca not in word2index.keys():
                word2index[voca] = len(word2index)
        return word2index

    def one_hot_encoding(self, word, word2index) :
        one_hot_vector = [0] * (len(word2index))
        index = word2index[word]
        one_hot_vector[index] = 1
        return one_hot_vector
    
    def set_data(self, tokens, word2index, window_size = 1) :
        X = []
        Y = []

        #윈도우 사이즈만큼 tokens에서 주변 원핫벡터 가져오기
        for token_idx in range(len(tokens)) :
            for i in range(1, window_size+1) : #윈도우 사이즈만큼 전방탐색
                if token_idx - i >= 0 :
                    X.append(self.one_hot_encoding(tokens[token_idx - i],word2index))
                    Y.append(self.one_hot_encoding(tokens[token_idx],word2index))
            for j in range(1, window_size+1) : #윈도우 사이즈만큼 후방탐색
                if token_idx + j <= len(tokens) -1 :
                    X.append(self.one_hot_encoding(tokens[token_idx + j],word2index))
                    Y.append(self.one_hot_encoding(tokens[token_idx],word2index))
        return X, Y
    
    def init_weight(self, X, word2index):

        W_1 = np.random.rand(len(word2index), self.H)
        W_2 = np.random.rand(self.H, len(word2index))

        hidden_node_ls = np.dot(X, W_1)
        
        output_ls = np.dot(hidden_node_ls, W_2)
        output_ls = self.softmax(output_ls)
        
        return W_1, W_2, hidden_node_ls, output_ls

    def softmax(self, a):
        return np.exp(a) /np.sum(np.exp(a), axis=1, keepdims=True)#np.exp(a) / np.sum(np.exp(a))

    def weight_update(self, X, Y, W_1, W_2, hidden_node_ls, output_ls) :

        temp = np.dot((output_ls- Y), W_2.T)

        dW_1 = np.dot(np.array(X).T, temp) 
        dW_2 = np.dot(hidden_node_ls.T, (output_ls - Y))

        
        W_1 -= self.learning_rate * dW_1
        W_2 -= self.learning_rate * dW_2

        hidden_node_ls = np.dot(X, W_1)
        
        output_ls = np.dot(hidden_node_ls, W_2)
        output_ls = self.softmax(output_ls)

        loss = -np.multiply(Y, np.log(output_ls)).sum()

        return output_ls, W_1, W_2, hidden_node_ls, loss
    
    def optimize(self, doc, epoch = 10000) :
        tokens = self.tokenize(doc)
        word2index = self.word_indexing(tokens)
        X, Y = self.set_data(tokens, word2index, window_size = 1)

        W_1, W_2, hidden_node_ls, output_ls = self.init_weight(X, word2index)
        

        for i in range(epoch) :
    
            output_ls, W_1, W_2, hidden_node_ls, loss = self.weight_update(X, Y, W_1, W_2, hidden_node_ls, output_ls)
            self.loss_ls.append(loss)
        
        Y_hat_ls = np.zeros((len(output_ls), len(output_ls[0])))
        for idx, output in enumerate(output_ls) :
            one_hot_index = output.argsort()[-1] #최대값 인덱스
            Y_hat_ls[idx][one_hot_index] = 1
        
        print(Y_hat_ls)
        print(Y)
        print("loss : {}".format(loss))

    def loss_graph(self):
        plt.plot(self.loss_ls)
        plt.title('Loss')
        plt.show()

        
doc = 'you will never know untill you try'
test = Word2Vec_CBow()
test.optimize(doc, 1000)
test.loss_graph()


