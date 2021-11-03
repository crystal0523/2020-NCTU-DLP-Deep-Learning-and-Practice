
import pandas as pd
from torch.utils import data
from function import word2tensor, idx2onehot
from torch.utils.data import DataLoader as torchLoader
from torch.utils.data import Dataset 

char_size = 28
MAX_LENGTH = 20


class Loader(data.Dataset):
    def __init__(self):
        
        Data = pd.read_csv('train.txt', sep = ' ', header = None)
        Data.columns = ['sp', 'tp', 'pg', 'p']#label
        self.data = []
        self.type_size=len(Data.columns)
        print(len(Data))
        for idx,column in enumerate(Data):
            
           for k in range(len(Data)):
               self.data.append(Data[[column]].iloc[k].tolist() + [idx] )#each word put in one list and compose the big list
              
        print("> Found {} pairs...".format(len(self.data))) # 1227*4
        print(self.data[0])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #print(self.data[index][0],self.data[index][1])
        return word2tensor(self.data[index][0]),idx2onehot(self.data[index][1],self.type_size)
Train=Loader()
print('len:',Train.__len__())
print(Train.__getitem__(5))

class Test_Loader(data.Dataset):
    def __init__(self):
        
        test = [[0, 3],
                 [0, 2],
                 [0, 1],
                 [0, 1],
                 [3, 1],
                 [0, 2],
                 [3, 0],
                 [2, 0],
                 [2, 3],
                 [2, 1]]
        
        Data = pd.read_csv('test.txt', sep = ' ', header = None)
        self.data = []
        
        for k in range(len(Data)):
            self.data.append(Data.iloc[k].tolist() + test[k])
        #print(self.data)
        print("> Found {} pairs...".format(len(self.data)))
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        print(self.data[index][0],self.data[index][1])
        return word2tensor(self.data[index][0]), word2tensor(self.data[index][1]), idx2onehot(self.data[index][2],4) , idx2onehot(self.data[index][3], 4)
Test=Test_Loader()
print(Test.__getitem__(5))

class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, num_workers=1):

        self.dataset = torchLoader(
             dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

if __name__ == "__main__":
    
    Train=Loader()
    Test=Test_Loader()
    
    traindata=DataLoader(dataset=Train,batch_size=1,shuffle=True)
    
    testdata=DataLoader(dataset=Test,batch_size=1,shuffle=False)