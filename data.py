
# import system libs
import os
from PIL import Image

# import data handling tools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# import Pytorch Libraries
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def get_df():
    # unzip the data
    if not os.path.exists('lung_colon_image_set'):
        # unzip archive.zip
        import zipfile
        with zipfile.ZipFile('archive.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        
    # Generate data paths with labels
    data_dir = 'lung_colon_image_set'
    filepaths = []
    labels = []

    folds = os.listdir(data_dir)
    for fold in folds:
        foldpath = os.path.join(data_dir, fold)
        flist = os.listdir(foldpath)

        for f in flist:
            f_path = os.path.join(foldpath, f)
            filelist = os.listdir(f_path)

            for file in filelist:
                fpath = os.path.join(f_path, file)
                filepaths.append(fpath)

                if f == 'colon_aca':
                    labels.append('Colon Adenocarcinoma')

                elif f == 'colon_n':
                    labels.append('Colon Benign Tissue')

                elif f == 'lung_aca':
                    labels.append('Lung Adenocarcinoma')

                elif f == 'lung_n':
                    labels.append('Lung Benign Tissue')

                elif f == 'lung_scc':
                    labels.append('Lung Squamous Cell Carcinoma')

    # Concatenate data paths with labels into one dataframe
    Fseries = pd.Series(filepaths, name= 'filepaths')
    Lseries = pd.Series(labels, name='labels')
    df = pd.concat([Fseries, Lseries], axis= 1)
    return df

# strat = df['labels']
# train_df, dummy_df = train_test_split(df,  train_size= 0.8, shuffle= True, random_state= 123, stratify= strat)

# # valid and test dataframe
# strat = dummy_df['labels']
# valid_df, test_df = train_test_split(dummy_df,  train_size= 0.5, shuffle= True, random_state= 123, stratify= strat)


def load_data_for_pytorch(df, batch_size):
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Creating a custom dataset
    class ImageDataset(torch.utils.data.Dataset):
        def __init__(self, dataframe, transform=None):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_name = self.dataframe.iloc[idx, 0]
            image = Image.open(img_name).convert('RGB')
            label = self.dataframe.iloc[idx, 1]
            
            if self.transform:
                image = self.transform(image)
            
            return image, label

    # Split the dataframe into train, valid, test
    strat = df['labels']
    train_df, dummy_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=123, stratify=strat)
    valid_df, test_df = train_test_split(dummy_df, train_size=0.5, shuffle=True, random_state=123, stratify=dummy_df['labels'])

    # Instantiate datasets
    train_dataset = ImageDataset(train_df, transform=transform)
    valid_dataset = ImageDataset(valid_df, transform=transform)
    test_dataset = ImageDataset(test_df, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader

if __name__ == "__main__":
    df = get_df()
    train_loader, valid_loader, test_loader = load_data_for_pytorch(df, 16)
    for data in train_loader:
        print(data)
        break
