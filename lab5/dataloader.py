# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:13:07 2020

@author: crystal
"""


import PIL.Image as Image
import json
from torch.utils import data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt


image_size = 64

class iclevrDataSet(data.Dataset):
    def __init__(self):
        with open('train.json', 'r') as f:
            file_dict = json.load(f)

        self.length = len(file_dict)
        self.img_name = list(file_dict.keys())
        self.labels = list(file_dict.values())
      
        self.transformations = transforms.Compose([transforms.Resize((64, 64)),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                                   ])#H*W*C=C*H*W
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        img_path = './iclevr/' + self.img_name[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transformations(img)
        #print(img.shape)
        labels = self.labels[index]
        label = []
        with open('objects.json', 'r') as f:
            obj_dict = json.load(f)

        for i in labels:
            label.append(obj_dict[i])

        labels = torch.zeros(24)
        for i in label:
            labels[i] = 1.0

        return img, labels
    
#train=iclevrDataSet()
#print(train.__getitem__(5))