# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 14:33:25 2020

@author: crystal
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import dataloader as dl
import matplotlib.pyplot as plt

#HYPER PARAMETER
lr= 0.01     
epochs = 600
BATCH_SIZE= 256

train_data, train_label, test_data, test_label = dl.read_bci_data()

def DataPreprocess(train_data, train_label, test_data, test_label):
    
    train_data=torch.FloatTensor(train_data)
    train_label=torch.LongTensor(train_label) #data轉為tensor形式
    
    test_data=torch.FloatTensor(test_data)
    test_label=torch.LongTensor(test_label)
    
    torch_dataset_train = Data.TensorDataset(train_data, train_label)
    
    train_loader = Data.DataLoader(
    dataset=torch_dataset_train,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=0,              # subprocesses for loading data
    )
    torch_dataset_test = Data.TensorDataset(test_data, test_label)
    
    test_loader = Data.DataLoader(
    dataset=torch_dataset_test,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=True,               # random shuffle for training
    num_workers=0,              # subprocesses for loading data
    )

    return train_loader,test_loader

train_loader,test_loader=DataPreprocess(train_data, train_label, test_data, test_label)

class DeepConvNet(nn.Module):
    def __init__(self,activation_function):
        super(DeepConvNet, self).__init__() #搭建網路起手式
        #input: 1*2*750
        activations=nn.ModuleDict([['ELU',nn.ELU(alpha=1.0)],['ReLU',nn.ReLU()],['LeakyReLU',nn.LeakyReLU()]])
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=25, kernel_size=(1,5),stride=(1,1),padding=(0,25),bias=False),
            #output channel: 提取16個特徵輸出到下一層
                
            )
       
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=(2,1),stride=(1,1),bias=False),
            nn.BatchNorm2d(25, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 4), padding=0 ),
            nn.Dropout(p=0.5) #neuron隨機屏蔽
            )
       
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=25, out_channels=50, kernel_size=(1,5),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(50, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.MaxPool2d(kernel_size=(1,2), stride=(1, 8), padding=0 ),
            nn.Dropout(p=0.5)
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=50, out_channels=100, kernel_size=(1,5),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(100, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.MaxPool2d(kernel_size=(1,2), stride=(1, 8), padding=0 ),
            nn.Dropout(p=0.5)
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels= 100, out_channels=200, kernel_size=(1,5),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(200, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.MaxPool2d(kernel_size=(1,2), stride=(1, 8), padding=0 ),
            nn.Dropout(p=0.5)
            )
        #full-connected layer
        self.out= nn.Sequential(
            nn.Linear(400,2,bias= True)
            )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)   
        x = self.conv4(x)
        x = self.conv5(x) 
        x= x.view(x.size(0),-1) #多維度的tensor展平成一維
        x = self.out(x)
        return x  
    


ELU_train_losses=[]
ELU_train_accuracy=[]
ELU_test_losses=[]
ELU_test_accuracy=[]

ReLU_train_losses=[]
ReLU_train_accuracy=[]
ReLU_test_losses=[]
ReLU_test_accuracy=[]

LeakyReLU_train_losses=[]
LeakyReLU_train_accuracy=[]
LeakyReLU_test_losses=[]
LeakyReLU_test_accuracy=[]

def Result_acc(ELU_train_accuracy,ELU_test_accuracy,ReLU_train_accuracy,ReLU_test_accuracy,LeakyReLU_train_accuracy,LeakyReLU_test_accuracy):
    max_ELU_acc=0
    max_ReLU_acc=0
    max_LeakyReLU_acc=0
    for i in range(epochs):
        
        if ELU_test_accuracy[i] > max_ELU_acc:
            max_ELU_acc= ELU_test_accuracy[i]
            
        if ReLU_test_accuracy[i] > max_ReLU_acc:
            max_ReLU_acc= ReLU_test_accuracy[i]    
            
        if LeakyReLU_test_accuracy[i] > max_LeakyReLU_acc:
            max_LeakyReLU_acc= LeakyReLU_test_accuracy[i]    
            
    plt.plot(ELU_train_accuracy, label='ELU_train_accuracy',color='b')
    plt.plot(ELU_test_accuracy, label='ELU_test_accuracy', color= 'g')
    plt.plot(ReLU_train_accuracy, label='ReLU_train_accuracy',color= 'r')
    plt.plot(ReLU_test_accuracy, label='ReLU_test_accuracy',color= 'c')
    plt.plot(LeakyReLU_train_accuracy, label='LeakyReLU_train_accuracy',color= 'm')
    plt.plot(LeakyReLU_test_accuracy, label='LeakyReLU_test_accuracy',color= 'y')
    plt.title('Activation function comparision(DeepConvNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()  
    plt.savefig('DeepConvNetacc.png')
    return max_ELU_acc,max_ReLU_acc,max_LeakyReLU_acc

torch.manual_seed(42)
'''
def restore_net():
   
    net = torch.load('DeepConvNet.pkl')
    net.eval()
    total_test = 0
    correct_test = 0
    criterion= nn.CrossEntropyLoss()
     
    for i, (x_data, x_label) in enumerate(test_loader):   
        
            data, labels = Variable(x_data), Variable(x_label)
            if torch.cuda.is_available(): # converting the data into GPU format
                net = net.cuda()
                criterion = criterion.cuda()
                data = data.cuda()
                labels = labels.cuda()  
                
            output = net(data)      
            loss = criterion(output, labels)  
            predicted=torch.max(output,1)[1]
            total_test += len(labels)
            correct_test += (predicted == labels).float().sum()      
    test_accuracy = 100 * correct_test / float(total_test)
    print('test Loss: {:.5f}'.format(loss.item()), "testing accuracy: %.2f %%" % (test_accuracy))
'''    
def train():
    model_save=0
    
    for activation_function in range(3):
        
        activation_functions_list = ['ELU', 'ReLU', 'LeakyReLU']
        net = DeepConvNet(activation_functions_list[activation_function])
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(),lr) 
        
        if torch.cuda.is_available(): # converting the data into GPU format
            net = net.cuda()
            criterion = criterion.cuda()
     
        for epoch in range(epochs):
    
            correct_train = 0
            total_train = 0
    
            for i ,(x_train, x_label)in enumerate(train_loader):#i for iteration
        
                train, train_labels = Variable(x_train), Variable(x_label)
        
                if torch.cuda.is_available(): 
                    train = train.cuda()
                    train_labels = train_labels.cuda()    
        
                optimizer.zero_grad() # clearing the Gradients of the model parameters
            
                output_train = net(train)
                train_loss = criterion(output_train, train_labels)
            
                train_loss.backward()#反向傳播求梯度
                optimizer.step()#更新所有參數
                predicted=torch.max(output_train,1)[1]
                total_train += len(train_labels)
                correct_train += (predicted == train_labels).float().sum()
    
            accuracy1 = 100 * correct_train / float(total_train)
        
            correct_test = 0
            total_test = 0
        
            for i, (y_test, y_label) in enumerate(test_loader):
        
                test, test_labels = Variable(y_test), Variable(y_label)
        
                if torch.cuda.is_available():
                    test = test.cuda()
                    test_labels = test_labels.cuda()    
            
                output_test=net(test)
        
                test_loss = criterion(output_test, test_labels)
                predicted=torch.max(output_test,1)[1]
                total_test += len(test_labels)
                correct_test += (predicted == test_labels).float().sum()
        
            accuracy2 = 100 * correct_test / float(total_test)
            
            '''if accuracy2 > model_save:
                model_save= accuracy2
                torch.save(net ,'DeepConvNet.pkl')
                #torch.save(net.state_dict(), 'net_params.pkl') 
                print('model save')'''
                
              
            if activation_function % 3 == 0:
                ELU_train_accuracy.append(accuracy1)
                ELU_test_accuracy.append(accuracy2)
              
            elif activation_function % 3 == 1:
                ReLU_train_accuracy.append(accuracy1)
                ReLU_test_accuracy.append(accuracy2)
          
            elif activation_function % 3 == 2:
                LeakyReLU_train_accuracy.append(accuracy1)
                LeakyReLU_test_accuracy.append(accuracy2)
           
            
            if epoch % 10 == 0:
                print('Epoch: ',epoch,'train Loss: {:.5f}'.format(train_loss.item()), "train accuracy: %.2f %%" % (accuracy1))
                print('Epoch: ',epoch,'test Loss: {:.5f}'.format(test_loss.item()), "test accuracy: %.2f %%" % (accuracy2))
    x,y,z=Result_acc(ELU_train_accuracy,ELU_test_accuracy,ReLU_train_accuracy,ReLU_test_accuracy,LeakyReLU_train_accuracy,LeakyReLU_test_accuracy)
    print("ELU test acc: %.2f %%" % (x), "ReLU test acc: %.2f %%" % (y),"LeakyReLU test acc: %.2f %%" % (z) )
    
    
train()
#restore_net()



 

