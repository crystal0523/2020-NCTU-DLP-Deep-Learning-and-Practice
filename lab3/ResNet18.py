# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 02:41:29 2020

@author: crystal
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torchvision import models
from dataloader import RetinopathyLoader 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.utils.model_zoo as model_zoo
import itertools
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EPOCHS = 10
BATCH_SIZE = 4

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'
}

path='/home/ubuntu/lab3/data/data/'


#path='C:/Users/crystal/Desktop/crystal/nctu/DL/lab3/data/'

Retin=RetinopathyLoader(path,'train')
Retin1=RetinopathyLoader(path,'test')

train_loader=Data.DataLoader(Retin,batch_size= BATCH_SIZE,shuffle= True, num_workers=2)
test_loader=Data.DataLoader(Retin1,batch_size= BATCH_SIZE,shuffle= True, num_workers=2)

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.stride= stride
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=5):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], 2)
        self.layer3 = self.make_layer(block, 256, layers[2], 2)
        self.layer4 = self.make_layer(block, 512, layers[3],2)
        self.avg_pool = nn.AvgPool2d(7,stride=1)
        self.fc = nn.Linear(51200, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels * block.expansion):
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion ))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * ResidualBlock.expansion
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1) #展平tensor
        out = self.fc(out)
        return out


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(ResidualBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        
    return model


# plot acc figure
def result(train_acc,test_acc,state):
    plt.figure()
    
    if state== 'w/o_trained':
        best_acc=0
        for i in range(EPOCHS):
            if test_acc[i]>best_acc:
                best_acc= test_acc[i]
        print('w/o_trained best test acc: ',best_acc)
        plt.plot(train_acc,label='train(w/o pretraining)',color='b')
        plt.plot(test_acc,label='test(w/o pretraining)',color='r')
        plt.xlabel ( "epoch" ) 
        plt.ylabel( "Accuracy(%)" ) 
        plt.title ( "ResNet18" ) 
    else:
        best_acc=0
        for i in range(EPOCHS):
            if test_acc[i]>best_acc:
                best_acc= test_acc[i]
        print('with_trained best test acc: ',best_acc)
        
        plt.plot(train_acc,label='train(with pretraining)',color='y')
        plt.plot(test_acc,label='test(with pretraining)',color='g')
    
        plt.legend()
        plt.savefig('resnet18_acc.png')
    
    

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):

  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized ResNet18 confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=45)
  plt.yticks(tick_marks, classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
  plt.figure()
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('ResNet18.png')


torch.manual_seed(42)
train_acc=[]
test_acc=[]
y_target=[]
y_predict=[]

def train(model,state,learning_rate = 1e-4 ):
    best_acc = 0.0
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(EPOCHS):
        
        #model.train()
        correct=0
        total=0     
        
        for i, (images, labels) in enumerate(train_loader):
            
            #print(images.shape,labels.shape)
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        train_acc.append(100 * correct / total)
        print ("Epoch [{}/{}],  train_loss: {:.4f}".format(epoch+1, EPOCHS, loss.item()))
        print('train accuracy{} %'.format(100 * correct / total))
             
        with torch.no_grad():
            correct = 0
            total = 0
            
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                model.eval()
                outputs = model(images)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                labels = labels.cpu().data.numpy().argmax()
                predicted = predicted.cpu().data.numpy().argmax()
                #print(labels,predicted)
                y_target.append(labels)
                y_predict.append(predicted)
                acc=100*correct / total
                
            test_acc.append(acc)
            
            print ("Epoch [{}/{}],  test_loss: {:.4f}".format(epoch+1, EPOCHS, loss.item()))
            print('test accuracy{} %'.format(100 * correct / total))
            
            if acc> best_acc:
                best_acc = acc
                print('model save')
                if state == 'w/o_trained':
                    torch.save({'state_dict': model.state_dict()}, 'checkpoint_resnet18.pth.tar')
                    
                else:
                    torch.save({'state_dict': model.state_dict()}, 'checkpoint_resnet18_pre.pth.tar')
                    
           
    # Decay learning rate 
        '''if (epoch+1) % 2 == 0:
            curr_lr /= 3
            update_lr(optimizer, curr_lr)
            '''
    result(train_acc,test_acc,state) 
    cm = confusion_matrix(y_target, y_predict)
    plot_confusion_matrix(cm, list(range(5)))
    print('training compelete')
    

def test(model,state):
    
    if state == 'w/o_trained':
        
        checkpoint = torch.load('checkpoint_resnet18.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])#
        model.eval()
        
    else:
       
        checkpoint = torch.load('checkpoint_resnet18_pre.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])#
        model.eval()
        
    with torch.no_grad():
        best_acc=0
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            acc=100 * correct / total
                 
        print('test accuracy{:.4f} %'.format(acc))

   

if __name__ == "__main__":
    
    model = resnet18(False)
    
    #print(model)
    model.cuda()
    #train(model,'w/o_trained')
    test(model,'w/o_trained')

    train_acc=[]
    test_acc=[]
    y_target=[]
    y_predict=[]
    model_pre = torchvision.models.resnet18(pretrained=True)
    model_pre.fc.out_features = 5
    model_pre.cuda()
    #train(model_pre,'with_trained',1e-5)
    test(model_pre,'with_trained')
   
    
