# -*- coding: utf-8 -*-
"""
Created on Sun Aug  2 11:36:41 2020

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

EPOCHS = 8
BATCH_SIZE = 4


path='/home/ubuntu/lab3/data/data/'


#path='C:/Users/crystal/Desktop/crystal/nctu/DL/lab3/data/'

Retin=RetinopathyLoader(path,'train')
Retin1=RetinopathyLoader(path,'test')

train_loader=Data.DataLoader(Retin,batch_size= BATCH_SIZE,shuffle= True, num_workers=2)
test_loader=Data.DataLoader(Retin1,batch_size= BATCH_SIZE,shuffle= True, num_workers=2)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
#Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 conv
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 conv
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion*100, num_classes)

        # kaiming weight normal after default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        

    # construct layer/stage conv2_x,conv3_x,conv4_x,conv5_x
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # when to need downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # inplanes expand for next block
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # suggest for adaptive pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
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
        plt.title ( "ResNet50" ) 
    else:
        best_acc=0
        for i in range(EPOCHS):
            if test_acc[i]>best_acc:
                best_acc= test_acc[i]
        print('with_trained best test acc: ',best_acc)
        
        plt.plot(train_acc,label='train(with pretraining)',color='y')
        plt.plot(test_acc,label='test(with pretraining)',color='g')
    
        plt.legend()
        plt.savefig('resnet50_acc.png')
    
    

def plot_confusion_matrix(cm, classes,normalize=True,title='Confusion matrix',cmap=plt.cm.Blues):
  plt.figure()
  if normalize:
      cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
      print("Normalized ResNet50 w/o pretrained confusion matrix")
  else:
      print('Confusion matrix, without normalization')

  #print(cm)

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
  
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.savefig('ResNet50.png')

train_acc=[]
test_acc=[]

torch.manual_seed(42)

def train(model,state,learning_rate = 1e-4):
    best_acc = 0.0
    y_target=[]
    y_predict=[]
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(EPOCHS):
        
        #model.train()
        correct=0
        total=0
        sum=0
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
            sum+=1
            print('running:',sum)
        #print('sum',sum)
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
                acc= 100*correct / total
                
            test_acc.append(acc)
            
            print ("Epoch [{}/{}],  test_loss: {:.4f}".format(epoch+1, EPOCHS, loss.item()))
            print('test accuracy{} %'.format(100 * correct / total))
            
            if acc> best_acc:
                best_acc = acc
                print('model save')
                if state == 'w/o_trained':
                    torch.save({'state_dict': model.state_dict()}, 'checkpoint_resnet50.pth.tar')
                   
                else:
                    torch.save({'state_dict': model.state_dict()}, 'checkpoint_resnet50_pre.pth.tar')
                    
           
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
        
        checkpoint = torch.load('checkpoint_resnet50.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
        
    else:
       
        checkpoint = torch.load('checkpoint_resnet50_pre.pth.tar')
        model.load_state_dict(checkpoint['state_dict'])
       
    
    with torch.no_grad():
        correct = 0
        total = 0
        best_acc=0
        
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            acc=100 * correct / total
            if acc>best_acc:
                best_acc=acc
        print('test accuracy{:.4f} %'.format(best_acc))
       
        

if __name__ == "__main__":
    
    model = resnet50(False)
    print(model)
    model.cuda()
    #train(model,'w/o_trained')
    '''test(model,'w/o_trained')

    train_acc=[]
    test_acc=[]

    model_pre = torchvision.models.resnet50(pretrained=True)
    model_pre.fc.out_features = 5
    model_pre.cuda()
    #train(model_pre,'with_trained',1e-5)
    test(model_pre,'with_trained')'''
   
    