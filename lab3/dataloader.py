import pandas as pd
#from torch.utils import data
import torch.utils.data as data
import numpy as np
#from torchvision import transforms
import torchvision.transforms as transforms
import cv2 
from PIL import Image

#吃csv資料為每個image標上號碼和label,尚未對應至圖片
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)#刪除數組形狀中的單維度條目
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
a,b=getData('train')
#print(a)

class RetinopathyLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            root (string): Root path of the dataset.
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)#吃csv資料為每個image標上號碼和label,尚未對應至圖片
        self.mode = mode
        #print(self.img_name)
        print("> Found %d images..." % (len(self.img_name)))

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpeg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpeg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        #print(self.img_name[index])
        path=self.root+self.img_name[index]+'.jpeg'
        img = Image.open(path)
        #print(type(img))
        transform2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),#影象一半的概率翻轉，一半的概率不翻轉
            #transforms.Resize((128, 128)),
            transforms.ToTensor(),## range [0, 255] -> [0.0,1.0]
            transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))])#image = (image - mean) / std, mean,std:0.5,0.5 歸一化到[-1.0, -1.0]
        img=transform2(img) #Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
        label=self.label[index]
    
        return img, label
#path='/home/ubuntu/lab3/data/data/'
#path='C:/Users/crystal/Desktop/crystal/nctu/DL/lab3/data/'
#Retin=RetinopathyLoader(path,'train')
#Retin.__getitem__(5)