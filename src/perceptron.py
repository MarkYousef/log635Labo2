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
        self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
       

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

        m = Y.shape[0]

        self.W1 = self.W1 - self.alpha* (capDelta1/m)
        self.W2 = self.W2 - self.alpha* (capDelta2/m)

    def loss(self, a3, Y):
       m = Y.shape[0]
       loss = (1/m)*np.sum(np.multiply(Y, np.log(a3)) + np.multiply((1-Y), np.log(1-a3)))
       return loss      

     #fonction tiré du document Perceptron.ipynb fournis
    def train(self,X,Y, n, eta, epsilon, training_images, training_labels):
    
         """  train_errors.append(0)
        
          for i in range(n):  
              xi = training_images[i]
              xi.resize(784, 1)
            
            # Compute activation for the example xi
              score = np.dot(W,xi)
            
              best_label = score.argmax(axis=0)
            
              if(best_label != training_labels[i]):
                  train_errors[epoch] = train_errors[epoch] + 1
                
          epoch = epoch + 1    
        
        # loop to update the weights
          for i in range(n):
              xi = training_images[i]
              xi.resize(784, 1)
            
              u = np.array(u_function(np.dot(W,xi)))
            
              dxi = np.zeros((1,10)).T
              dxi[training_labels[i]] = 1
              dxi_minus_u = np.subtract(dxi, u)
            
              p = np.multiply(dxi_minus_u, np.transpose(xi))
              W = W + eta * p
            
        # while loop terminates when algorithm converges or when number of epoch is 100
        # when training data is quite high, eta is 1 and epsilon is 0 then
        # there are high chances of algorithm not getting converged
        # making while loop run infinitely, so max. epochs is set to 100
          if(train_errors[epoch - 1]/n <= epsilon or epoch==100):
              break """
         for i in range(iter):
            yPred = self.forward_prog(self,X)
            loss = self.loss(self,yPred,Y )

            self.backward_prog(self,Y)

            #print something ... 

         
       
       

        