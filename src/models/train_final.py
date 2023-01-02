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
from dataloader_v2 import PreTrainLoader, PreValLoader
from torchvision.transforms.functional import InterpolationMode
import logging
import sys
import math
from tqdm import tqdm
from model import ViTSimilarModel
from model_v2 import ViTSimilarModel_v2
from siamese import Siamese
from contrastive import ContrastiveLoss
from super_contrastive_model import Siamese_v2
from loss import supervisedContrastiveLoss
from collections import defaultdict
from torch.cuda import amp
from train_supervised import CFG
from dataloader_supervised import SimilarityValDataset, SimilarityDataset

class FinalTrainWrapper:
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
            compose = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)])
            self.train_dataset = SimilarityDataset(compose)
            # compose_val = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC)])
            self.val_dataset = SimilarityValDataset(compose)
            
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
            self.optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
        if 'scheduler' in kwargs:
            self.scheduler = kwargs['scheduler']
        else:
            self.scheduler = lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr)
            
        if 'device' in kwargs:
            self.device = kwargs['device']
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_loader = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=5)
        self.val_loader = DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=True,num_workers=6)

    def trainModel(self):
        start = time.time()
        # best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = np.inf
        history = defaultdict(list)
        scaler = amp.GradScaler()
        dataloaders = {'train':self.train_loader,'val':self.val_loader}
        dataset_sizes = {'train':len(self.train_dataset),'val':len(self.val_dataset)}
        self.model.to(CFG.device)
        for step, epoch in enumerate(range(1,self.num_epochs+1)):
            print('Epoch {}/{}'.format(epoch, self.num_epochs))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train','val']:
                if(phase == 'train'):
                    self.model.train() # Set model to training mode
                else:
                    self.model.eval() # Set model to evaluation mode
                
                running_loss = 0.0
                running_correct=0.0
                
                # Iterate over data
                for img1,img2,labels in tqdm(dataloaders[phase]):
                    img1 = img1.to(CFG.device)
                    img2 = img2.to(CFG.device)
                    labels = labels.to(CFG.device)
                    # labels = labels.view(labels.size()[0])
                    # print(labels)
                    # l=input()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        with amp.autocast(enabled=True):
                            outputs = self.model(img1,img2)
                            loss = self.criterion(outputs, labels)
                            loss = loss / CFG.n_accumulate
                            
                        
                        # backward only if in training phase
                        if phase == 'train':
                            scaler.scale(loss).backward()

                        # optimize only if in training phase
                        if phase == 'train' and (step + 1) % CFG.n_accumulate == 0:
                            scaler.step(self.optimizer)
                            scaler.update()
                            self.scheduler.step()
                            
                            # zero the parameter gradients
                            self.optimizer.zero_grad()


                    running_loss += loss.item()*img1.size(0)
                    y_pred = outputs.round()
                    running_correct += y_pred.eq(labels).sum()
                epoch_loss = running_loss/dataset_sizes[phase]  
                epoch_acc = running_correct/dataset_sizes[phase]          
                history[phase + ' loss'].append(epoch_loss)
                history[phase + ' acc'].append(epoch_acc)

                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
                
                print('{} acc: {:.4f}'.format(
                    phase, epoch_acc))
                # # deep copy the model
                # if phase=='valid' and epoch_loss <= best_loss:
                #     best_loss = epoch_loss
                #     best_model_wts = copy.deepcopy(model.state_dict())
                #     # PATH = f"Fold{fold}_{best_loss}_epoch_{epoch}.bin"
                #     torch.save(model.state_dict(), 'best_model.bin')

            print()

        end = time.time()
        time_elapsed = end - start
        print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
            time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
        print("Best Loss ",best_loss)

        # load best model weights
        # model.load_state_dict(best_model_wts)
        return self.model, history   