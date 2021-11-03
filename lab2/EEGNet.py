# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 23:37:59 2020

@author: crystal
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import dataloader as dl
import matplotlib.pyplot as plt

#HYPER PARAMETER
lr= 0.001     
epochs = 600
BATCH_SIZE= 64

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

class EEGNet(nn.Module):
    def __init__(self,activation_function):
        super(EEGNet, self).__init__() #搭建網路起手式
        #input: 1*2*750
        activations=nn.ModuleDict([['ELU',nn.ELU(alpha=1.0)],['ReLU',nn.ReLU()],['LeakyReLU',nn.LeakyReLU()]])
     
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False),
            nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True)    
            )
       
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1),stride=(1,1),groups=16, bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0 ),
            nn.Dropout(p=0.25) #neuron隨機屏蔽
            )
       
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
            activations[activation_function],
            nn.AvgPool2d(kernel_size=(1,8), stride=(1, 8), padding=0 ),
            nn.Dropout(p=0.25)
            )
        #full-connected layer
        self.out= nn.Sequential(
            nn.Linear(736,2,bias= True)
            )
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)   
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
    plt.title('Activation function comparision(EEGNET)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.legend()
    plt.show()  
    plt.savefig('acc.png')
    return max_ELU_acc,max_ReLU_acc,max_LeakyReLU_acc

torch.manual_seed(42)
'''
def restore_net():
   
    net = torch.load('EEGNet.pkl')
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

def restore_params():
    # 新建 net3
   
    class net1(nn.Module):
        def __init__(self):
            super(net1, self).__init__() #搭建網路起手式
        #input: 1*2*750
       # activations=nn.ModuleDict([['ELU',nn.ELU(alpha=1.0)],['ReLU',nn.ReLU()],['LeakyReLU',nn.LeakyReLU()]])
     
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(1,51),stride=(1,1),padding=(0,25),bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True)    
                )
       
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,1),stride=(1,1),groups=16, bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
                nn.ReLU(1.0),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0 ),
                nn.Dropout(p=0.25) #neuron隨機屏蔽
                )
       
            self.conv3 = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine= True, track_running_stats= True),
                nn.ReLU(1.0),
                nn.AvgPool2d(kernel_size=(1,8), stride=(1, 8), padding=0 ),
                nn.Dropout(p=0.25)
                )
            #full-connected layer
            self.out= nn.Sequential(
                nn.Linear(736,2,bias= True)
                )
            
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)   
            x= x.view(x.size(0),-1) #多維度的tensor展平成一維
            x = self.out(x)
            return x  
    
    net=net1()
    # 将保存的参数复制到 net1
    net.load_state_dict(torch.load('net.pkl'))
    net.eval()
    total_test = 0
    correct_test = 0
    criterion= nn.CrossEntropyLoss()
   
    for i, (x_data, x_label) in enumerate(test_loader):   
        
            data, labels = Variable(x_data,requires_grad=False), Variable(x_label,requires_grad=False)
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
    print('total_test:',total_test,'correct_test:',correct_test)
    test_accuracy = 100 * correct_test / float(total_test)
    print('test Loss: {:.5f}'.format(loss.item()), "testing accuracy: %.2f %%" % (test_accuracy))
'''    
   
def train():
    model_save=0
    
    for activation_function in range(3):
        
        activation_functions_list = ['ELU', 'ReLU', 'LeakyReLU']
        net = EEGNet(activation_functions_list[activation_function])
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
        
                test, test_labels = Variable(y_test,requires_grad=False), Variable(y_label,requires_grad=False)
        
                if torch.cuda.is_available():
                    test = test.cuda()
                    test_labels = test_labels.cuda()    
            
                output_test=net(test)
        
                test_loss = criterion(output_test, test_labels)
                predicted=torch.max(output_test,1)[1]
                total_test += len(test_labels)
                correct_test += (predicted == test_labels).float().sum()
        
            accuracy2 = 100 * correct_test / float(total_test)
            '''
            if accuracy2 > model_save:
                model_save= accuracy2
                torch.save(net ,'EEGNet.pkl')
                torch.save(net.state_dict(), 'net.pkl')
                
                print("Model's state_dict:")
                for param_tensor in net.state_dict():
                    print(param_tensor, "\t", net.state_dict()[param_tensor])
                print('model save') '''
              
            if activation_function % 3 == 0:
                ELU_train_accuracy.append(accuracy1)
                ELU_test_accuracy.append(accuracy2)
              
            elif activation_function % 3 == 1:
                ReLU_train_accuracy.append(accuracy1)
                ReLU_test_accuracy.append(accuracy2)
          
            elif activation_function % 3 == 2:
                LeakyReLU_train_accuracy.append(accuracy1)
                LeakyReLU_test_accuracy.append(accuracy2)
           
            
            if epoch % 50 == 0:
                print('Epoch: ',epoch,'train Loss: {:.5f}'.format(train_loss.item()), "train accuracy: %.2f %%" % (accuracy1))
                print('Epoch: ',epoch,'test Loss: {:.5f}'.format(test_loss.item()), "test accuracy: %.2f %%" % (accuracy2))
           
    x,y,z=Result_acc(ELU_train_accuracy,ELU_test_accuracy,ReLU_train_accuracy,ReLU_test_accuracy,LeakyReLU_train_accuracy,LeakyReLU_test_accuracy)
    print("ELU test acc: %.2f %%" % (x), "ReLU test acc: %.2f %%" % (y),"LeakyReLU test acc: %.2f %%" % (z) )
    
    
train()

#restore_net()
#restore_params()



 