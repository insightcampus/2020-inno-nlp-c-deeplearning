class XOR() :
    def __init__(self, X, Y, h) :
        import numpy as np
        self.X = X
        self.Y = Y
        self.hidden_node_number = h


    def _init_weights(self) :
        import random
        ###매트릭스 생성

        H = self.hidden_node_number
        input_layer_matrix = np.random.rand(2, H)
        b1_matrix = np.random.rand(H, 1)
        hidden_layer_matrix = np.random.rand(H, 1)
        b2_matrix = np.random.rand(1, 1)

        return input_layer_matrix, hidden_layer_matrix, b1_matrix, b2_matrix
        
    def sigmoid(self, x):
        return 1 / (1 +np.exp(-x))

    def _feedfoward(self,input_layer_matrix, hidden_layer_matrix, b1_matrix, b2_matrix) : #H, Y_hat 계산 및 sigmoid화
        
        #hidden_node
        hidden_node_list = b1_matrix + np.dot(input_layer_matrix.T, self.X)
        hidden_node_list = self.sigmoid(hidden_node_list)

        #output node
        output_node_list = self.sigmoid(b2_matrix + np.dot(hidden_layer_matrix.T, hidden_node_list))
        output_node_list = self.sigmoid(output_node_list)
        
        return hidden_node_list, output_node_list

    def _eval_loss (self, output_node_list):
        import math
        loss = -1 * 1/self.X.shape[1] * np.sum(self.Y * np.log(output_node_list)+ (1 - self.Y) * np.log(1 - output_node_list))
        return loss

    def _gradients (self, input_layer_matrix, hidden_layer_matrix, b1_matrix, b2_matrix, output_node_list, hidden_node_list, leaning_rate = 0.1):
        
        leaning_rate = leaning_rate
        cost = output_node_list - self.Y

        gradient_w2 = np.dot(hidden_node_list, cost.T) 
        
        gradient_b2 = 1. / self.Y.shape[1] * np.sum(cost, axis = 1, keepdims= True)
        
        gradient_h = np.dot(hidden_layer_matrix, cost)

        z1 = gradient_h * hidden_node_list * (1 - np.array(hidden_node_list))
       
        gradient_w1 = np.dot(self.X, z1.T)
        
        gradient_b1 = 1./self.Y.shape[1] * np.sum(z1, axis = 1, keepdims= True)
                
        #가중치 학습
        
        input_layer_matrix += -leaning_rate * gradient_w1 # w1 업데이트

        hidden_layer_matrix += -leaning_rate * gradient_w2 # w2 업데이트

        b1_matrix += -leaning_rate * gradient_b1 # b1 업데이트

        b2_matrix += -leaning_rate * gradient_b2 # b2 업데이트

        return input_layer_matrix, hidden_layer_matrix, b1_matrix, b2_matrix


    def optimize (self, learning_rate = 0.1, epoch = 2000) :
        
        input_layer_matrix, hidden_layer_matrix, b1_matrix, b2_matrix = self._init_weights() #w1, w2, b1, b2 가중치 초기화

        for count in range(epoch) :
            #hidden_node와 Y_hat 계산 후 sigmoid화
            hidden_node_list = b1_matrix + np.dot(input_layer_matrix.T, self.X)
            hidden_node_list = self.sigmoid(hidden_node_list)

            output_node_list = b2_matrix + np.dot(hidden_layer_matrix.T, hidden_node_list)
            output_node_list = self.sigmoid(output_node_list)

            #w1, w2, b1, b2 가중치 학습 
            a = learning_rate
            

            cost = output_node_list - self.Y

            gradient_w2 = np.dot(hidden_node_list, cost.T) 
            
            gradient_b2 = 1. / self.Y.shape[1] * np.sum(cost, axis = 1, keepdims= True)
            
            gradient_h = np.dot(hidden_layer_matrix, cost)

            z1 = gradient_h * hidden_node_list * (1 - np.array(hidden_node_list))
        
            gradient_w1 = np.dot(self.X, z1.T)
            
            gradient_b1 = 1./self.Y.shape[1] * np.sum(z1, axis = 1, keepdims= True)
                    
            #가중치 학습
            
            input_layer_matrix -= a * gradient_w1 # w1 업데이트

            hidden_layer_matrix -= a * gradient_w2 # w2 업데이트

            b1_matrix -= a * gradient_b1 # b1 업데이트

            b2_matrix -= a * gradient_b2 # b2 업데이트

            if count%1000 == 0 :
                # print("gradient_w2 : {}".format(gradient_w2))
                # print("gradient_b2 : {}".format(gradient_b2))
                # print("gradient_w1 : {}".format(gradient_w1))
                # print("gradient_b1 : {}".format(gradient_b1))
                print("\n**********************\n")
                print("loss : {}".format(self._eval_loss(output_node_list)))
                print(output_node_list)
                print(output_node_list.round(0))
                print("\n**********************\n")

        return output_node_list.round(3)

import numpy as np

X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

test = XOR(X, Y, 3)

predicts = test.optimize(0.1, 100000)
print("Y hat : {}".format(predicts))
print("predicts : {}".format([1 if y > 0.5 else 0 for y in predicts[0]]))