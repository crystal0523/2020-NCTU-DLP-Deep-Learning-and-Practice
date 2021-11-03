
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from function import EOS_token, SOS_token


class Encoder(nn.Module):
    #encoder train each vector之 mean, var
    def __init__(self, input_size, latent_size, condition_size, linear_size, device="cpu"):
        super(Encoder, self).__init__()
        self.latent_size = latent_size-condition_size
        self.device = device
        self.embedding = nn.Embedding(input_size, latent_size)
        self.lstm = nn.LSTM(latent_size, latent_size)
        self.linear_means = nn.Linear(latent_size, linear_size)
        self.linear_log_var = nn.Linear(latent_size, linear_size)


    def forward(self, x, c1):
        h0 = torch.zeros(1, 1, self.latent_size)
        c1 = c1.view(1, 1, -1)       
        hidden = torch.cat((h0, c1), dim=-1)
        cur=torch.zeros_like(hidden)  
        embedded = self.embedding(x).unsqueeze(1)
        
        #print(embedded.shape, hidden.shape)
        _, (hidden ,_ ) = self.lstm(embedded, (hidden,cur))

        means = self.linear_means(hidden)
        log_vars = self.linear_log_var(hidden)

        return means, log_vars


class Decoder(nn.Module):

    def __init__(self, latent_size, output_size, condition_size, linear_size, tf_ratio, MAX_LENGTH, device="cpu"):
        super(Decoder, self).__init__()
        self.device = device
        self.tf_ratio = tf_ratio
        self.MAX_LENGTH = MAX_LENGTH

        self.l1 = nn.Linear(linear_size + condition_size, latent_size)
        self.embedding = nn.Embedding(output_size, latent_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(latent_size, latent_size)
        self.fc = nn.Linear(latent_size, output_size)

    def forward(self, z, c, target=None):
        z = torch.cat((z.view(1, 1, -1), c.view(1, 1, -1)), dim=-1)
        #print(z.shape)
        hidden = self.l1(z)
        cur=torch.zeros_like(hidden)
        MAX_LENGTH = target.size(1) if torch.is_tensor(
            target) else self.MAX_LENGTH
        use_teacher_forcing = torch.is_tensor(
            target) and random.random() < self.tf_ratio
        
      
        decoder_input = torch.Tensor([SOS_token]).long()
        decoder_output = []
        #print('decoder input',decoder_input)
        for index in range(MAX_LENGTH):
            #print('decoder input',index , decoder_input)
            embedded = self.embedding(decoder_input).view(1, 1, -1)
            output = self.relu(embedded)
            output, (hidden , cur) = self.lstm(output, (hidden,cur))
            output = self.fc(output)
            #print('output',output.shape)
            

            decoder_output.append(output[0])
        
            if use_teacher_forcing:
                #print('targer 1:', target[index+1])
                decoder_input = target[: ,index]                
               # print('decoder 1',decoder_input)
            else:
                topi = output.topk(1)[1]
                
                decoder_input = topi.squeeze().detach()  # detach from history as input
                #print('decoder 2:',decoder_input)
            if decoder_input.item() == EOS_token:
                break

        return torch.stack(decoder_output)
    
    
class seq2seqCVAE(nn.Module):

    def __init__(self, encoder_sizes, latent_size, decoder_sizes, condition_size, tf_ratio=0, MAX_LENGTH=20, device="cpu"):
        
        super(seq2seqCVAE, self).__init__()

        assert condition_size > 0 #斷定condiion
        self.linear_size = 32
        self.device = device
        self.encoder = Encoder(encoder_sizes, latent_size,
                               condition_size, self.linear_size, device)
        self.decoder = Decoder(latent_size, decoder_sizes,
                               condition_size, self.linear_size, tf_ratio, MAX_LENGTH, device)
    #reparameterization
    def forward(self, x, c1, c2,  target=None):

        means, log_var = self.encoder(x, c1)

        std = torch.exp(0.5 * log_var)
        eps = torch.randn(self.linear_size)#epsilon是一個從標準正態分佈（均值為0，方差為1，即高斯白噪聲）中抽取的一組隨機數
        #z: global latent sentence representation
        z = eps * std + means   
        #print(z.shape)

        output = self.decoder(z, c2, target)

        return output, means, log_var
    
    def inference(self, c):
        z = torch.randn(self.linear_size)
        output = [self.decoder(z, cc) for cc in c]
        return output

    def tr_ratio_anneal(self, new_tr_ratio):
        self.decoder.tf_ratio = new_tr_ratio
