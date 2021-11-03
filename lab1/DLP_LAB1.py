# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 22:45:50 2020

@author: crystal
"""
import numpy as np 
import matplotlib.pyplot as plt

def generate_linear(n=100):
    import numpy as np
    pts=np.random.uniform(0,1,(n,2))
    inputs=[]
    labels=[]
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance=(pt[0]-pt[1])/1.414
        if pt[0]>pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs),np.array(labels).reshape(n,1)

def generate_XOR_easy():
    import numpy as np
    inputs=[]
    labels=[]
    for i in range (11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)
        if 0.1*i==0.5:
            continue
        inputs.append([0.1*i,1-0.1*i])
        labels.append(1)
    return np.array(inputs),np.array(labels).reshape(21,1)

#x,y=generate_linear(n=100)
x,y=generate_XOR_easy()   

def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return x * (1 - x)
'''
def tanh(x):
    return (1.0 - np.exp(-2*x))/(1.0 + np.exp(-2*x))

def tanh_derivative(x):
    return (1 + x)*(1 - x)

'''
class NeuralNetwork:
    
    def __init__(self,x,y,epoch):
        self.x = x
        self.weights1= np.random.rand(x.shape[1],4)# 4 neuron in hidden layer
        self.weights2 = np.random.rand(4,4)
        self.weights3=  np.random.rand(4,1)
        self.y = y
        self.epoch=epoch
        self.loss=[]
        self.output = np. zeros(self.y.shape)
        
    def forward(self):
        self.H1 = np.dot(self.x, self.weights1)
        self.z1= sigmoid(self.H1)
        
        self.H2 = np.dot(self.z1, self.weights2)
        self.z2= sigmoid(self.H2)
        
        self.H3=np.dot(self.z2, self.weights3)
        self.output = sigmoid(self.H3)
        
        #return self.output
        
    def backprop(self,lr=0.1):
        
        error = self.y-self.output
        self.dw3=error*sigmoid_derivative(self.output)
        
        error=np.dot(error,self.weights3.T)
        self.dw2=error*sigmoid_derivative(self.z2)
        
        error=np.dot(error,self.weights2.T)
        self.dw1=error*sigmoid_derivative(self.z1)
        
        
        self.weights1 += lr*np.dot(self.x.T,self.dw1)
        self.weights2 += lr*np.dot(self.z1.T,self.dw2)
        self.weights3 += lr*np.dot(self.z2.T,self.dw3)
        #print('weights update: ',self.weights1,'adjust: ',self.dw1,'\n')
              
    def train(self):
        for i in range(self.epoch):
            self.forward()
            self.backprop()
            loss=np.mean(np.square(self.y - self.output))#MSE
            self.loss.append(loss)
            pred_y=np.round(self.output)
            if (i+1)%100==0:
                print("Epoch:" +str(i+1)+" Loss: " + str(loss))
                print("Accuracy :", float((np.sum(pred_y ==self.y) /self.y.shape[0])*100),'%')
        self.plot_loss()
       # print(pred_y)
        return pred_y
    
    def plot_loss(self):      
        plt.plot(self.loss)
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Loss curve")
        plt.show()

  
def show_result(x,y,pred_y):
    import matplotlib.pyplot as plt
    plt.subplot(1,2,1)
    plt.title('Ground truth',fontsize=18)
    for i in range (x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.subplot(1,2,2)
    plt.title('Predict result',fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()  
    
nn=NeuralNetwork(x,y,10000)           
pred_y=nn.train()
show_result(x,y,pred_y)