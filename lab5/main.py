# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 14:39:44 2020

@author: crystal
"""

import PIL.Image as Image
import json
from torch.utils import data
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
import torchvision.models as models
import torchvision
import matplotlib.pyplot as plt
from dataloader import iclevrDataSet
from model import Generator, Discriminator
from evaluator import evaluation_model

# hyperparameters 
batch_size = 32
z_dim = 128
lr = 2e-4
n_epoch = 1000


# model
G = Generator(z_dim).cuda()
D = Discriminator(4).cuda()
print(G)
print(D)
# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr,betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr,betas=(0.5, 0.999))

# dataset
dataset = iclevrDataSet()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test conditions
with open('objects.json', 'r') as file:
    obj_dict = json.load(file)
    #print('obj')
    #print(obj_dict)
    
with open('test.json','r') as file:
    test_dict = json.load(file)
    #print('test')
    #print(test_dict)
    n_test = len(test_dict)
    #print(n_test)
    

test_c = torch.zeros(n_test, 24)
#print('test_c',test_c.shape)#32*24, n_test=32,initialized with 0


for i in range(n_test):
    for condition in test_dict[i]:
        test_c[i, obj_dict[condition]] = 1.
        

test_z = torch.randn(n_test, z_dim)#從normal distribution中隨機抽取噪聲
test_c = test_c.cuda()
#print('shape:',test_c.shape)
test_z = test_z.cuda()
eval_model = evaluation_model()
def train():
    best_score = 0
    for epoch in range(n_epoch):
        G.train()
        D.train()
        for i, (imgs, c) in enumerate(dataloader):
            imgs = imgs.cuda()
            c = c.cuda()#c: 128張圖片(batch_size)的condtion(multi-hot encoding)
            #print(c[0])
            #print(c.shape)#128*24
            bs = imgs.size(0) # batch size
            #print(bs)#128
            # label        
            real_label = torch.ones(bs)
            fake_label = torch.zeros(bs)
            real_label = real_label.cuda()
            fake_label = fake_label.cuda()
        
            """ Train Discriminator """#discriminator每訓練 6次generator訓練一次, 用固定的generator來 train discriminator
            if (i+1)%6==0 or i==0:
                opt_D.zero_grad()
                z = torch.randn(bs, z_dim).cuda()#z:128*128
                fake_imgs = G(z, c)#c:condition

                # discriminator
                #detach: fix 網路參數
                real_logit = D(imgs.detach(), c)
                fake_logit = D(fake_imgs.detach(), c)
            
                # compute loss
                real_loss = criterion(real_logit, real_label)
                fake_loss = criterion(fake_logit, fake_label)
           
                loss_D = (real_loss + fake_loss)/2 #?

                # update model
                loss_D.backward()
                opt_D.step()

            """ Train Generator """
        
            opt_G.zero_grad()
            z = torch.randn(bs, z_dim).cuda()
            fake_imgs = G(z, c)

            # discriminator
            fake_logit = D(fake_imgs, c)
        
            # compute loss
            loss_G = criterion(fake_logit, real_label)
        
            # update model
            loss_G.backward()
            opt_G.step()

    
            print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    
        G.eval()
        with torch.no_grad():
            gen_imgs = G(test_z, test_c) 

        score = eval_model.eval(gen_imgs, test_c)
        print(f'\nScore: {score:.2f}')
        if score > best_score:
            print('parameters saved!\n')
            torch.save(G.state_dict(), 'netG_weight.pth')
            torch.save(D.state_dict(), 'netD_weight.pth')
            torchvision.utils.save_image(gen_imgs, 'result1.png', nrow=8, normalize=True)
            best_score = score

        # show generated image
        #grid_img = torchvision.utils.make_grid(gen_imgs.cpu(), nrow=8, normalize=True)
        #plt.figure(figsize=(8, 4))
        #plt.imshow(grid_img.permute(1, 2, 0))
        #plt.show()
def test():    
    #test_z = torch.randn(32, z_dim)
    test_z = torch.load('new_test_z.pt')
    test_c = test_c.cuda()
    test_z = test_z.cuda()
    eval_model = evaluation_model()

    G = Generator(z_dim).cuda()
    G.load_state_dict(torch.load('netG_weight.pth'))

    with torch.no_grad():
        gen_imgs = G(test_z, test_c) 
        #print('shape:',test_c.shape)
        score = eval_model.eval(gen_imgs, test_c)
        print(f'\nScore: {score:.2f}')

    # show generated image
    grid_img = torchvision.utils.make_grid(gen_imgs.cpu(),'final_result1.png' ,nrow=8, normalize=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(grid_img.permute(1, 2, 0))
    #plt.show()


if __name__ == "__main__":
    
    train()
    test()