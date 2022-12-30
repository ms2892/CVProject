import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt 
import time
import os 
from torch.utils.data import DataLoader
import copy
# from torchsummary import summary
from transformers import ViTModel
from torchvision.transforms.functional import InterpolationMode
from dataloader import TinyImageNetLoader

class ViTSimilarModel(nn.Module):
    
    def __init__(self):
        super(ViTSimilarModel,self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-large-patch32-384')
        
        
    def forward(self,x):
        
        # Fix the Forward Pass
        # print(x.shape)
        img1 = x[:,0,:,:,:]
        img2 = x[:,1,:,:,:]
        # print(img1.shape)
        out1 = self.vit(img1)
        out1 = out1.last_hidden_state
        out1 = torch.flatten(out1,start_dim=1)
        
        out2 = self.vit(img2)
        out2 = out2.last_hidden_state
        out2 = torch.flatten(out2,start_dim=1)
        
        cosine_sim = nn.CosineSimilarity(dim=1)
        out3 = cosine_sim(out1,out2)
        out3 = torch.reshape(out3,(-1,1))
        # print(out3.shape)
        # x=input()
        out3.add_(1)
        out3.div_(2)
        # print(out3.shape)
        # t=input()
        return out3

if __name__=='__main__':
    dataset = TinyImageNetLoader(transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC)]))
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
    data_iter = iter(dataloader)



    inputs, labels = next(data_iter)
    img1 = inputs[0]
    img2 = inputs[1]
    print(img1.shape)
    model = ViTSimilarModel()
    print(model(inputs))
    print(labels)
    