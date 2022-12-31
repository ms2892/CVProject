# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from glob import glob
import random
import pickle
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
from os.path import exists

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    
    
    logger = logging.getLogger(__name__)
    logger.info('Creating the Training Dataset')
    
    
    if not exists('../../data/processed/pairs_v2.txt'):
    
        train_files = glob('../../data/raw/tiny-imagenet-200/train/*/images/*')
        class_labels = glob('../../data/raw/tiny-imagenet-200/train/*')
        classes={}
        for i in range(len(class_labels)):
            class_labels[i]=class_labels[i].replace('\\','/')
            split = class_labels[i].split('/')
            classes[split[6]]=[]
        # print(classes)
        for i in range(len(train_files)):
            train_files[i]=train_files[i].replace('\\','/')
            split = train_files[i].split('/')
            classes[split[6]].append(train_files[i])
        labels={'sim':[],'oth':[]}
        flag_sim=True
        flag_oth=True
        cnt_sim=0
        cnt_oth=0
        sim_size=5000
        class_size = sim_size/200
        with tqdm(total=2*sim_size) as pbar:
            generated=set()
            for class_ in classes.keys():
                cnt=0
                while(cnt<class_size):
                    ele = random.choices(classes[class_],k=2)
                    if ele[0]==ele[1]:
                        continue
                    else:
                        if (ele[0],ele[1]) not in generated:
                            generated.add((ele[0],ele[1]))
                            labels['sim'].append(ele)
                            cnt_sim+=1
                            cnt+=1
                            pbar.update(1)
            
            while(flag_oth):
                ele = random.choices(train_files,k=2)
                cls1= ele[0].split('/')[6]
                cls2= ele[1].split('/')[6]
                if cls1!=cls2:
                    if (ele[0],ele[1]) not in generated:
                        generated.add((ele[0],ele[1]))
                        cnt_oth+=1
                        pbar.update(1)
                        labels['oth'].append(ele)
                        if cnt_oth==sim_size:
                            break

            # print(ele)/
            # t=input()
        print(len(labels['sim']),len(labels['oth']))
        print(labels['sim'][0])
        with open('../../data/processed/pairs_v2.txt','wb') as handle:
            pickle.dump(labels,handle)
        
        
        
        logger.info('Training Pair Successfully Created and stored at ' + '../../data/processed/pairs.txt')
    else:
        logger.info('Training Pair txt file already exists')
        
        
    logger.info('Creating Validation Dataset')
    generated=set()
    
    if not exists('../../data/processed/val_v2.txt'):
        
        val_files = glob('../../data/raw/tiny-imagenet-200/val/images/*')
        for i in range(len(val_files)):
            val_files[i] = val_files[i].replace('\\','/')
            
        labels={'sim':[],'oth':[]}
        flag_sim=True
        flag_oth=True
        cnt_sim=0
        cnt_oth=0
        val_annotations = pd.read_csv('../../data/raw/tiny-imagenet-200/val/val_annotations.txt',delimiter='\t')
        
        class_fname = defaultdict(list)
        
        lst = val_annotations.iloc[:,:].values
        
        for i in lst:
            # print(i)
            class_fname[i[1]].append('../../data/raw/tiny-imagenet-200/val/images/'+i[0])

        with tqdm(total=0.4*sim_size) as pbar:
            while(flag_oth):
                ele = random.choices(val_files,k=2)
                split1 = ele[0].split('/')
                fname1 = split1[7]
                split2 = ele[1].split('/')
                fname2 = split2[7]
                
                class1 = val_annotations.loc[val_annotations['FileName']==fname1]
                class1 = class1.iloc[:,1].values
                class1 = class1[0]
                # print(class1)
                # x=input()
                class2 = val_annotations.loc[val_annotations['FileName']==fname2]
                class2 = class2.iloc[:,1].values
                class2 = class2[0]

                if class1!=class2 and flag_oth:
                    if (fname1,fname2) not in generated:
                        generated.add((fname1,fname2))
                        labels['oth'].append(ele)
                        cnt_oth+=1
                        pbar.update(1)
                        if cnt_oth==0.2*sim_size:
                            flag_oth=False
            classes = val_annotations[['FileName','Class']]
            classes = classes.Class.unique()
            classes = list(classes)
            # print(classes)
            # x=input()
            while(flag_sim):
                rndm_class = random.choice(classes)
                temp = class_fname[rndm_class]
                ele = random.choices(class_fname[rndm_class],k=2)
                split1=ele[0].split('/')
                split2=ele[1].split('/')
                fname1=split1[7]
                fname2=split2[7]
                
                if fname1!=fname2:
                    if (fname1,fname2) not in generated:
                        generated.add((fname1,fname2))
                        labels['sim'].append(ele)
                        cnt_sim+=1
                        pbar.update(1)
                        if cnt_sim==0.2*sim_size:
                            flag_sim=False


            # print(ele)/
            # t=input()
        print(len(labels['sim']),len(labels['oth']))
        print(labels['sim'][0])
        with open('../../data/processed/val_v2.txt','wb') as handle:
            pickle.dump(labels,handle)
    else:
        logger.info('Validation pairs already exists. Skipping this step')
        
    logger.info('Successfully generated the datasets')
    # print(val_files)
    
    
    # print(train_files)
    
    # logger = logging.getLogger(__name__)
    # logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()
