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
from dataloader import TinyImageNetLoader, TinyImageNetValLoader
from contrastive import ContrastiveLoss
import torch.nn.functional as F

class ViTSimilarModel_v2(nn.Module):
    
    def __init__(self):
        super(ViTSimilarModel_v2,self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-large-patch32-384')
        for params in self.vit.parameters():
            params.requires_grad=False
        self.linear = nn.Linear(148480,1024)
        self.linear2 = nn.Linear(1024,128)
        self.linear3 = nn.Linear(128,2)
        self.relu = nn.ReLU()
        
        
    def forward(self,input1,input2):
        out1 = self.vit(input1)
        out2 = self.vit(input2)
        out1 = out1.last_hidden_state
        out2 = out2.last_hidden_state
        
        
        
        out1 = out1.view(out1.size()[0], -1)
        out2 = out2.view(out2.size()[0],-1)
        
        # print(out1.shape)
        
        out1 = self.linear(out1)
        out1 = self.relu(out1)
        out1 = self.linear2(out1)
        out1 = self.relu(out1)
        out1 = self.linear3(out1)
        
        out2 = self.linear(out2)
        out2 = self.relu(out2)
        out2 = self.linear2(out2)
        out2 = self.relu(out2)
        out2 = self.linear3(out2)
        return out1,out2

if __name__=='__main__':
    dataset = TinyImageNetValLoader(transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC)]))
    dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
    data_iter = iter(dataloader)



    img1,img2, labels = next(data_iter)
    # img1 = inputs[0]
    criterion = ContrastiveLoss()

    # img2 = inputs[1]
    print(img1.shape)
    model = ViTSimilarModel_v2()
    output1,output2 = model(img1,img2)
    eucledian_distance = F.pairwise_distance(output1, output2)
    print(eucledian_distance)
    print(labels)
    