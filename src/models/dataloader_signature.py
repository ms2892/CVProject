import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math
from glob import glob
import pickle
from torchvision.transforms.functional import InterpolationMode
import cv2
import pandas as pd

class SignatureDataset(Dataset):

    def __init__(self, transform=None):
        pth = pd.read_csv('../../data/testing/test_data.csv')
        self.y = pth.iloc[:,-1].values
        self.y = 1- self.y
        self.pre_path = '../../data/testing/test/'
        self.x = pth.iloc[:,:-1].values
        # print(self.x.shape,self.y)
        self.transform = transform

    def __getitem__(self,index):
        img1, img2 = self.get_images(self.x[index])
        return img1,img2,self.y[index]

    def get_images(self,paths):
        pth1 = self.pre_path + paths[0]
        pth2 = self.pre_path + paths[1]
        
        img1 = cv2.imread(pth1)
        img2 = cv2.imread(pth2)
        
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        # img1 = cv2.resize(img1,(64,64))
        # img2 = cv2.resize(img2,(64,64))
        
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)
        
        # print(img1.shape,img1.shape)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        # print(img1.shape,img2.shape)
        return img1,img2
    
    def __len__(self):
        return len(self.x)




if __name__=='__main__':
    
    compose = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)])
    dataset = SignatureDataset(compose)
    
    t=dataset[0]