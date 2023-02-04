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
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class PreTrainLoader(Dataset):
    
    def __init__(self,pth,transforms=None):
        super(PreTrainLoader,self).__init__()
        self.pth = glob(pth+'/*/images/*')
        self.classes = glob(pth+'/*') 
        self.cls = {}
        for i in range(len(self.classes)):
            self.classes[i] = self.classes[i].replace('\\','/')
            self.classes[i] = self.classes[i].split('/')[-1]
            self.cls[self.classes[i]] = i
        for i in range(len(self.pth)):
            self.pth[i]=self.pth[i].replace('\\','/')
        # print(self.pth)
        self.transforms = transforms
        
    def __getitem__(self, index):
        img = cv2.imread(self.pth[index])
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(img.shape)
        # img = np.moveaxis(img,-1,0)
        # print(img.shape)
        img = img.astype(np.uint8)
        # print(img.shape)
        if self.transforms!=None:
            img = self.transforms(img)
        class_ = self.pth[index].split('/')[6]
        return img, self.cls[class_]
        
    def __len__(self):
        return len(self.pth)
        
class PreValLoader(Dataset):
    def __init__(self,transform=None):
        super(PreValLoader,self).__init__()
        with open('../../data/processed/val_contrast.txt','rb') as handle:
            input_label = pickle.loads(handle.read())
        # print(input_label)
        self.x = [i for i in input_label.keys()]
        self.y = [input_label[i] for i in self.x]
        self.transform = transform
        
    def __getitem__(self, index):
        img = cv2.imread(self.x[index])
        
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # print(img.shape)
        # img = np.moveaxis(img,-1,0)
        # print(img.shape)
        img = img.astype(np.uint8)
        # print(img.shape)
        if self.transform!=None:
            img = self.transform(img)
        return img, self.y[index]
    
    def __len__(self):
        return len(self.x)
        
if __name__=='__main__':
    compose = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.RandAugment()])
    
    dataset = PreValLoader(transform=compose)
    print(dataset[4])