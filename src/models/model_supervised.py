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
from train_model import CFG
import timm
import cv2
from dataloader_supervised import SimilarityDataset, SimilarityValDataset

class SimilarityModel(nn.Module):
    def __init__(self):
        super(SimilarityModel,self).__init__()
        model = timm.create_model(CFG.model_name, pretrained=True)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, CFG.embedding_size)

        out = model(torch.randn(1, 3, CFG.img_size, CFG.img_size))
        model.load_state_dict(torch.load('model_weights.pth'))
        self.model = model
        for params in self.model.parameters():
            params.requires_grad=False
        # print(self.model)

    def forward(self,input1,input2):
        out1 = self.model(input1)
        out2 = self.model(input2)
        cosine = nn.CosineSimilarity(dim=1)
        out3 = cosine(out1,out2)
        out3 = out3.view(out3.shape[0],1)
        # out3 = torch.sigmoid(out3)
        # print(out3.shape)
        return out3


if __name__=='__main__':
    x=SimilarityModel()
    compose = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224),interpolation=InterpolationMode.BICUBIC)])
    dataset = SimilarityValDataset(compose)
    dataloader = DataLoader(dataset,batch_size=32, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.to(device)
    t=0
    x.eval()
    # for img1,img2,labels in dataloader:
    #     img1 = img1.to(device)
    #     img2 = img2.to(device)
    #     labels = labels.to(device)
    #     output = x(img1,img2)
    #     # print(output)
    #     output = output>0.21
    #     t += (output==labels).sum()
    #     # print(labels)
        

    #     # p=input()
    # print(t/len(dataset))

    data_iter = iter(dataloader)

    input1 ,input2, labels = next(data_iter)
    input1 = input1.to(device)
    input2 = input2.to(device)
    labels = labels.to(device)
    output = x(input1,input2)
    output = output>0.2

    for i in range(input1.size()[0]):
        img1 = input1[i]

        img1 = transforms.ToPILImage()(img1)
        img1 = np.asarray(img1)
        img1 = cv2.resize(img1,(64,64))
        img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)

        img2 = input2[i]

        img2 = transforms.ToPILImage()(img2)
        img2 = np.asarray(img2)
        img2 = cv2.resize(img2,(64,64))
        img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)

        # img = np.moveaxis()
        # print(img1.shape)

        f, axarr = plt.subplots(1,2)
        f.suptitle(f'Label: {str(labels[i].item())}, Predicted: {str(output[i].item())}', fontsize=14, fontweight='bold')

        axarr[0].imshow(img1)
        axarr[1].imshow(img2)
        plt.show()
        # l=input()