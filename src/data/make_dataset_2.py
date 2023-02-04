from glob import glob
import pandas as pd
import numpy as np
from collections import defaultdict
import pickle


val_files = glob('../../data/raw/tiny-imagenet-200/val/images/*')
for i in range(len(val_files)):
    val_files[i] = val_files[i].replace('\\','/')
    
val_annotations = pd.read_csv('../../data/raw/tiny-imagenet-200/val/val_annotations.txt',delimiter='\t')
files = glob('../../data/raw/tiny-imagenet-200/train/*')
for i in range(len(files)):
    files[i] = files[i].replace('\\','/')
    files[i] = files[i].split('/')[6]

lst = val_annotations.iloc[:,:].values
files.sort()
input_label = {}
for i in lst:
    if i[1] in files:
        input_label['../../data/raw/tiny-imagenet-200/val/images/'+i[0]] = files.index(i[1])
print(input_label)
with open('../../data/processed/val_contrast.txt','wb') as handle:
    pickle.dump(input_label,handle)