import torch
import torchvision
from torch.utils.data import Dataset,DataLoader
import torch.nn as nn
import numpy as np
import math

class TinyImageNetLoader(Dataset):
    
    def __init__(self,path,transform=None):
        