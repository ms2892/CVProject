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
import copy
from torch.utils.data import DataLoader
from dataloader import TinyImageNetLoader
from torchvision.transforms.functional import InterpolationMode
import logging
import sys
import math
logger = logging.getLogger(__name__)
from tqdm import tqdm



class TrainModelWrapper:
    # 
    # model,dataset,num_epochs,batch_size,optimizer,criterion,scheduler
    
    def __init__(self,model,criterion,**kwargs):
        self.model = model
        self.criterion = criterion
        if 'dataset' in kwargs:
            try:
                self.train_dataset = kwargs['dataset']['train']
                self.val_dataset = kwargs['dataset']['val']
            except:
                logging.info('Format of the dataset is not proper. Please return the dataset as a dictionary like \{\'train\': training_dataset, \'val\':validation_dataset \}')
                sys.exit(" ")
        else:
            compose = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC), transforms.RandAugment(num_ops=2,magnitude=5)])
            self.dataset = TinyImageNetLoader(compose)
            compose_val = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC)])
            
        if 'num_epochs' in kwargs:
            self.num_epochs = kwargs['num_epochs']
        else:
            self.num_epochs = 25
        if 'batch_size' in kwargs:
            self.batch_size = kwargs['batch_size']
        else:
            self.batch_size = 4
        if 'lr' in kwargs:
            self.lr = kwargs['lr']
        else:
            self.lr = 0.001    
        if 'optimizer' in kwargs:
            self.optimizer = kwargs['optimizer']
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(),lr=self.lr)
        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
        else:
            self.scheduler = lr_scheduler.StepLR(self.optimizer,step_size=7,gamma=0.1)
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True)
        
    def trainModel(self):
        total_train_iters = math.ceil(len(self.train_loader)/self.batch_size)
        total_val_iters = math.ceil(len(self.val_loader)/self.batch_size)
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            with tqdm(total=total_train_iters) as pbar:
                for inputs,labels in self.train_loader:
                    
                    
                    
                    
                    pbar.update(1)