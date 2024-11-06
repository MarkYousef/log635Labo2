import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
from tabulate import tabulate
import idx2numpy

class Perceptron:

    def __init__(self, alpha=0.1, iters=10):
        self.alpha = alpha
        self.iters = iters
        #faut voir quelle valeurs mettre. 
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        self.b1 = np.ones((self.hiddenLayerSize))

        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        self.b2 = np.ones((self.outputLayerSize))

    def sigmoid(z):
      return 1/(1 + np.exp(-z))

    def sigmoid_diff(f):
      return f * (1 - f)
    

    def sigmoid_Prime(z):
      return np.exp(-z)/((1+np.exp(-z))**2)
    
    def forward_prog(self, X):
       self.a1 = X
       self.z2 = np.dot(X, self.W1)
       self.a2 = self.sigmoid(self.z2)
       self.z3 = np.dot(self.a2, self.W2)
       self.a3 = self.sigmoid(self.z3) 
       return self.a3
    
    #fonction tiré du document Perceptron.ipynb fournis
    def predict(self, X):
        return self.forward_prog(X)
    
    def backward_prog(self, Y):
       
        delta3 = self.a3 - Y
        delta2 = np.dot(delta3,self.W2.T)*self.sigmoid_Prime(self.z2)

        capDelta1 = np.dot(self.a1.T,delta2)
        capDelta2 = np.dot(self.a2.T,delta3)

        m = self.a1.shape[0]

        self.W1 = self.W1 - self.alpha* (capDelta1/m)
        self.b2 = self.b2 - self.alpha * np.sum(delta3, axis=0)
        self.W2 = self.W2 - self.alpha* (capDelta2/m)
        self.b1 = self.b1 - self.alpha * (1./m) * np.sum(delta2, axis=0)

    def loss(self, a3, Y):
       m = Y.shape[0]
       loss = (1/m)*np.sum(np.multiply(Y, np.log(a3)) + np.multiply((1-Y), np.log(1-a3)))
       return loss      

    # Tiré du ficher FFNN.ipynb 
    def train(self,X,Y):
     for i in range(self.iters):
            y_pred = self.forward_prog(self,X)
            loss = self.loss(self,self.a3, y_pred)
            
            self.backward_prog(self, Y)

            if i == 0 or i ==self.iters-1:
                print(f"Iteration: {i+1}")
                print(tabulate(zip(X, y, [np.round(y_pred) for y_pred in self.A3] ), headers=["Input", "Actual", "Predicted"]))
                print(f"Loss: {loss}")                
                print("\n")
         
       
       

        