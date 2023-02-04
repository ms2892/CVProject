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
from dataloader_v2 import PreTrainLoader
from loss import supervisedContrastiveLoss


class Siamese_v2(nn.Module):
    def __init__(self):
        super(Siamese_v2,self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3,stride=1),
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            
            nn.Conv2d(96, 256, kernel_size=5,stride=1,padding=2),
            nn.ReLU(inplace=True),
            # nn.LocalResponseNorm(5,alpha=0.0001,beta=0.75,k=2),
            nn.MaxPool2d(3, stride=2),
            # nn.Dropout2d(p=0.3),

            nn.Conv2d(256,384 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(384,256 , kernel_size=3,stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),
            # nn.Dropout2d(p=0.3),
        )
        # Defining the fully connected layers
        self.fc1 = nn.Sequential(
            nn.Linear(173056, 1024),
            nn.ReLU(inplace=True),
            # nn.Dropout2d(p=0.5)
            )
        
    def forward(self,input1):
        output = self.cnn1(input1)
        output = output.view(output.size()[0], -1)
        # print(output.shape)
        
        output = self.fc1(output)
        
        # output2 = self.cnn1(input2)
        # output2 = output2.view(output2.size()[0], -1)
        # output2 = self.fc1(output2)
        
        return output
    
if __name__=="__main__":
    dataset = PreTrainLoader('../../data/raw/tiny-imagenet-200/train',transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)]))
    # print(dataset[0])
    dataloader = DataLoader(dataset,batch_size=32,shuffle=True)
    data_iter = iter(dataloader)



    img, labels = next(data_iter)
    # print(img.shape)
    # img1 = inputs[0]
    # img2 = inputs[1]
    # print(img1.shape)
    model = Siamese_v2()
    output = model(img)
    # print(model(img))
    criterion = supervisedContrastiveLoss()
    l = criterion(output,labels)
    print(l)
    print(labels)