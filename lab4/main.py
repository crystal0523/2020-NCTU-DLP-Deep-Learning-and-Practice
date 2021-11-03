
import os
#import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from torch import optim
from Dataloader import Loader, Test_Loader, char_size, MAX_LENGTH , DataLoader
from function import (asMinutes, compute_bleu, get_output, idx2onehot,Gaussian_score,
                       tensor2word, timeSince, word2tensor)
from model import seq2seqCVAE

"""==============================================================================
The sample.py includes the following template functions:

1. Encoder, decoder
2. Training function
3. BLEU-4 score function

You have to modify them to complete the lab.
In addition, there are still other functions that you have to 
implement by yourself.

1. The reparameterization trick
2. Your own dataloader (design in your own way, not necessary Pytorch Dataloader)
3. Output your results (BLEU-4 score, words)
4. Plot loss/score
5. Load/save weights

There are some useful tips listed in the lab assignment.
You should check them before starting your lab.
================================================================================"""




device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
_SAVEPATH = _PATH+"save/"
if not os.path.isdir(_SAVEPATH):
    os.mkdir(_SAVEPATH)

bestscore = []

#----------Hyper Parameters----------#

hidden_size = 256
teacher_forcing_ratio = 1
KLD_weight = 0
KLW_max = 0.1
LR = 0.05
criterion = torch.nn.CrossEntropyLoss()
kl_anneal_type = ["logistic", "linear"]

plotdata = pd.DataFrame(
    columns=["tr_ratio", "KL_W", "loss", "CE_loss", "KL_loss"])
scores = pd.DataFrame(columns=["Bleu-4"])

blue=[]

def kl_anneal_function(anneal_function, step, k=0.2, epochs=30):

    if anneal_function == "logistic":
        return float(1/(1+np.exp(-k*(step-epochs/2))))
    elif anneal_function == "linear":
        return step/epochs * KLW_max


def loss_fn(output, target, mean, log_var):
    CE_Loss = 0
    #target=target.unsqueeze(1)
    #print(output.shape,target.shape)
    #output = output.unsqueeze(1)
    for i in range(output.size(0)):
        
        CE_Loss += criterion(output[i], target[i+1]) # crossentropy 計算loss
        
    #求N(mean, log變異數)和N(0,1)之KL Divergence
    #類別資訊 Y，希望同一個類的樣本都有一個專屬的均值 μ^Y（方差不變），μ^Y 讓模型自己訓練出來
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp()) 
    return CE_Loss / output.size(0), KLD / output.size(0)


def train(X, target, c1, c2, model, optimizer): 
    model.train()
    X = X.view(-1)
    
    #print('target2model',target)
    #print('target[:,1:]',target[: ,1:])
    output, means, log_var = model(X, c1, c2, target[ : ,1:]) #求出隱向量的mean var
    #print('output :', output.shape)
    CE_Loss, KLD = loss_fn(output, target.permute(1,0) , means, log_var)
    
    # loss sum= Reconstruction losses + KL divergence losses summed over all elements and batch
    
    loss = (CE_Loss + KLD_weight * KLD)

    optimizer.zero_grad()
    #KLD backward以使其值趨近0
    loss.backward()
    
    optimizer.step()

    return [loss.item(), CE_Loss.item(), KLD.item()]


def test():
    model.eval()

    print("Seq2seq Test")
    print("input\t\t\tGT\t\t\toutput\t\t\tbleu")
    avgbleu = 0
    for X, target, c1, c2 in testdata.dataset:
        print(X.shape)
        X = X.view(-1)
        #c = c.permute(1, 0, 2)
        pred = model(X, c1, c2)[0]
        output = get_output(pred)

        Wx, Wg, Wo = tensor2word(X), tensor2word(
            target.view(-1)), tensor2word(output)
    
        bleu = compute_bleu(Wo, Wg)
        print('Wx:',Wx,'Wg:',Wg,'Wo:',Wo,'bleu:',bleu,'\n')
        #print("{:<16}\t{:<16}\t{:<20}\t{:.4f}\n".format(Wx, Wg, Wo, bleu))
        avgbleu += bleu
    avgbleu /= 10
    print("avgbleu :", avgbleu)
        

    for i in range(100):
        c = idx2onehot(list(range(eng.type_size)), eng.type_size) #char 2 one-hot enocding
        output = model.inference(c)
        gens = [tensor2word(get_output(o)) for o in output]
        print(gens)
        print("g_scores :", Gaussian_score(gens))
        
    return avgbleu




def trainmodel():
    best_score=0

    global KLD_weight, teacher_forcing_ratio 

    
    optimizer = optim.SGD(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    for epoch in range(1,31):
        print("----------\nEpoch {}".format(epoch))
      
        #annealing function decide KLD_weight
        KLD_weight = kl_anneal_function(kl_anneal_type[1], epoch-1, epochs=30)
        
        if epoch %3 == 0 and epoch !=0 and epoch !=3: #revise
            teacher_forcing_ratio *= 0.85
            
        model.tr_ratio_anneal(teacher_forcing_ratio)
        scheduler.step()

        losses = [[], [], []]
        for X, c in traindata.dataset : #從traindata 提取 word 跟 condition
            #print('X target', X)
            #print(X.shape)
            c1=c
            c2=c
            loss = train(X, X, c1, c2, model, optimizer) 
            for i, l in enumerate(loss):
                losses[i].append(l)

        lo = np.mean(losses, axis=1)
        plotdata.loc[epoch-1] = [teacher_forcing_ratio,
                                 KLD_weight, lo[0], lo[1], lo[2]]
        print(plotdata.loc[epoch-1].to_frame().T)


        testscore = test()
        scores.loc[len(scores)] = testscore
        blue.append(testscore)
        if best_score < testscore:
    
            print("Saving model", testscore)
            state = {
                  "model": model.state_dict(),
                  "score": testscore,
                }
            torch.save(state, _SAVEPATH+"S2SCVAE_linear.t7")
            best_score = testscore
    plt.figure()
    scores.plot()
    plt.savefig(_PATH+"bleu4.png")
    plt.close()
    plt.figure()
    plotdata.plot()
    plt.savefig(_PATH+"all.png")
    plt.close()



def DEMO():
    if os.path.isfile(_SAVEPATH+"S2SCVAE_linear.t7"):
        print("Loading model")
        savedmodel = torch.load(_SAVEPATH+"S2SCVAE_linear.t7")
        model.load_state_dict(savedmodel["model"])
        test()
    else:
        print("No record")


if __name__ == "__main__":
    
    
    eng=Loader()
    Train=Loader()
    Test=Test_Loader()
    traindata=DataLoader(Train, batch_size=1,shuffle=True)
    testdata= DataLoader(Test, batch_size=1,shuffle=False)
    #print(traindata)
    #print(testdata)
    #condition_size: 標籤個數(條件)
    #latent_size=hidden_size
    model = seq2seqCVAE(char_size, hidden_size, char_size, condition_size=eng.type_size,
                        tf_ratio=teacher_forcing_ratio, device=device, MAX_LENGTH=MAX_LENGTH)
    #trainmodel()  #訓練model

    DEMO()
