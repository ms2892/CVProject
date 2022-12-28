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

class TinyImageNetLoader(Dataset):
    
    def __init__(self,transform=None):
        super(TinyImageNetLoader,self).__init__()
        with open('../../data/processed/pairs.txt','rb') as handle:
            self.pairs = pickle.loads(handle.read())
        self.x = [i for i in self.pairs['sim']]
        self.y = ['sim']*50000
        
        self.x += [i for i in self.pairs['oth']]
        self.y += ['oth']*50000
        self.transform = transform
        
    def __getitem__(self, index):
        img1, img2 = self.get_images(self.x[index])
        return (img1,img2), self.y[index]
        
    def get_images(self,ele):
        pth1 = ele[0]
        pth2 = ele[1]
        
        img1 = cv2.imread(pth1)
        img2 = cv2.imread(pth2)
        
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        
        img1 = np.moveaxis(img1,-1,0)
        img2 = np.moveaxis(img2,-1,0)
        
        img1 = torch.from_numpy(img1)
        img2 = torch.from_numpy(img2)
        
        img1 = img1.type(torch.uint8)
        img2 = img2.type(torch.uint8)
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        return img1,img2
    
if __name__=='__main__':
    compose = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC), transforms.RandAugment(num_ops=2,magnitude=0)])
    dataset = TinyImageNetLoader(compose)
    
    imgs, label = dataset[2000]
    
    img1 = imgs[0].numpy()
    img1 = np.moveaxis(img1,0,-1)
    img1 = cv2.cvtColor(img1,cv2.COLOR_RGB2BGR)
    
    img2 = imgs[1].numpy()
    img2 = np.moveaxis(img2,0,-1)
    img2 = cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
    
    
    
    cv2.imshow(label,img1)
    cv2.waitKey(0)
    cv2.imshow(label,img2)
    cv2.waitKey(0)