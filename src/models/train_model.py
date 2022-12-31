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
from dataloader import TinyImageNetLoader, TinyImageNetValLoader
from torchvision.transforms.functional import InterpolationMode
import logging
import sys
import math
from tqdm import tqdm
from model import ViTSimilarModel
from model_v2 import ViTSimilarModel_v2
from siamese import Siamese
from contrastive import ContrastiveLoss

logger = logging.getLogger(__name__)

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
            self.train_dataset = TinyImageNetLoader(compose)
            compose_val = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC)])
            self.val_dataset = TinyImageNetValLoader(compose_val)
            
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
            
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=5)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True,num_workers=6)
        
    def trainModel(self):
        total_train_iters = len(self.train_loader)
        total_val_iters = len(self.val_loader)
        self.model = self.model.to(self.device)
        # self.model= self.model.to(self.device)
        since = time.time()
        for epoch in range(self.num_epochs):
            print(f'Epoch {epoch+1}/{self.num_epochs}:')
            running_loss=0
            self.model.train()
            with tqdm(total=total_train_iters) as pbar:
                for img1,img2,labels in self.train_loader:
                    
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    labels = labels.to(self.device)
                    
                    output1,output2 = self.model(img1,img2)
                    
                    loss = self.criterion(output1,output2,labels)
                    running_loss+=loss.item()
                    
                    
                    
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    pbar.update(1)
            self.scheduler.step()
            epoch_loss = running_loss / len(self.train_dataset)
            print(f'Training Loss: {epoch_loss}')

            
            self.model.eval()
            running_loss=0
            with tqdm(total=total_val_iters) as pbar:
                for img1,img2, labels in self.val_loader:
                    img1 = img1.to(self.device)
                    img2 = img2.to(self.device)
                    labels = labels.to(self.device)
                    
                    output1,output2 = self.model(img1,img2)

                    loss = self.criterion(output1,output2,labels)
                    

                    
                    running_loss+= loss.item()
                    pbar.update(1)

            val_loss = running_loss/len(self.val_dataset)

            print(f'Validation Loss: {val_loss}')

        time_elapsed = time.time()-since
        print('---------------------------------------------------------------------')
        print(f'Training Completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s')
        return self.model           
        
        
if __name__=='__main__':
    model = ViTSimilarModel_v2()
    
    criterion = ContrastiveLoss()
    
    trainer = TrainModelWrapper(model,criterion,batch_size=8,num_epochs=10)
    best_model = trainer.trainModel()