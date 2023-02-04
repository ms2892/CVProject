import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses

class supervisedContrastiveLoss(nn.Module):
    def __init__(self,temperature=0.07):
        super(supervisedContrastiveLoss,self).__init__()
        self.temperature = temperature
        
    def forward(self,features,labels):
        feature_vectors_normalized = F.normalize(features, p=2, dim=1)
        logits = torch.div(
            torch.matmul(
                feature_vectors_normalized, torch.transpose(feature_vectors_normalized, 0, 1)
            ),
            self.temperature,
        )
        return losses.NTXentLoss(temperature=0.07)(logits, torch.squeeze(labels))
    
