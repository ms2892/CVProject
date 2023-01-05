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


class SimilarityDataset(Dataset):

    def __init__(self, transform=None):
        with open('../../data/processed/pairs_v3.txt','rb') as handle:
            self.pairs = pickle.loads(handle.read())
        self.x = [i for i in self.pairs['sim']]
        self.y = [[1.0]]*len(self.x)
        
        self.x += [i for i in self.pairs['oth']]
        self.y += [[0.0]]*len(self.y)
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

        self.transform = transform

    def __getitem__(self,index):
        img1, img2 = self.get_images(self.x[index])
        return img1,img2,self.y[index]

    def get_images(self,paths):
        pth1 = paths[0]
        pth2 = paths[1]
        img1 = cv2.imread(pth1)
        img2 = cv2.imread(pth2)

        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)


        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        if self.transform != None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2
    
    def __len__(self):
        return len(self.x)



class SimilarityValDataset(Dataset):

    def __init__(self, transform=None):
        with open('../../data/processed/val_v3.txt','rb') as handle:
            self.pairs = pickle.loads(handle.read())
        self.x = [i for i in self.pairs['sim']]
        self.y = [[1.0]]*len(self.x)
        
        self.x += [i for i in self.pairs['oth']]
        self.y += [[0.0]]*len(self.y)
        self.y = torch.from_numpy(np.array(self.y).astype(np.float32))

        self.transform = transform

    def __getitem__(self,index):
        img1, img2 = self.get_images(self.x[index])
        return img1,img2,self.y[index]

    def get_images(self,paths):
        pth1 = paths[0]
        pth2 = paths[1]
        img1 = cv2.imread(pth1)
        img2 = cv2.imread(pth2)
        img1 = img1.astype(np.uint8)
        img2 = img2.astype(np.uint8)

        if self.transform != None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1,img2
    
    def __len__(self):
        return len(self.x)



if __name__=='__main__':
    compose = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)])
    # train_dataset = SimilarityDataset(compose)
    # print(train_dataset[6000])


    val_dataset = SimilarityValDataset(compose)
    print(val_dataset[100])