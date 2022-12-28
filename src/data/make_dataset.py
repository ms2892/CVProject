# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from glob import glob
import random
import pickle
from tqdm import tqdm

# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
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
    
    with tqdm(total=100000) as pbar:
        while(flag_sim or flag_oth):
            ele = random.choices(train_files,k=2)
            split1 = ele[0].split('/')
            class1 = split1[6]
            split2 = ele[1].split('/')
            class2 = split2[6]
            
            if class1==class2 and flag_sim:
                labels['sim'].append(ele)
                pbar.update(1)
                cnt_sim+=1
                if cnt_sim==50000:
                    flag_sim=False
            elif class1!=class2 and flag_oth:
                labels['oth'].append(ele)
                cnt_oth+=1
                pbar.update(1)
                if cnt_oth==50000:
                    flag_oth=False
        # print(ele)/
        # t=input()
    print(len(labels['sim']),len(labels['oth']))
    print(labels['sim'][0])
    with open('../../data/processed/pairs.txt','wb') as handle:
        pickle.dump(labels,handle)
    
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
