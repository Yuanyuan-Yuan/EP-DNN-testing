import os
import sys
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
import torchvision.transforms as transforms

class CelebADataset(Dataset):
    def __init__(self, 
                img_dir='/export/d3/user/dataset/celeba_crop128/',
                name2id_path='/export/d3/user/dataset/celeba_crop128/CelebA_name_to_ID.json',
                id2name_path='/export/d3/user/dataset/celeba_crop128/CelebA_ID_to_name.json',
                split='train',
                img_size=64):
        assert split in ['train', 'val', 'test']
        self.transform = transforms.Compose([
                       transforms.Resize(img_size),
                       transforms.CenterCrop(img_size),
                       transforms.ToTensor()
               ])
        with open(name2id_path, 'r') as f:
            self.name2id = json.load(f)
        with open(id2name_path, 'r') as f:
            self.id2name = json.load(f)
        self.name_dir = img_dir + split + '/'
        self.name_list = sorted(os.listdir(self.name_dir))

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        im = Image.open(self.name_dir + self.name_list[index])
        tsr = self.transform(im)
        c = torch.LongTensor([0])
        return tsr, c.squeeze()


class ImageNetDataset(Dataset):
    def __init__(self,
                 image_dir='/export/d3/user/dataset/ImageNet/',
                 label2index_file='/export/d3/user/dataset/ImageNet/ImageNetLabel2Index.json',
                 split='train',
                 image_size=128):
        super(ImageNetDataset).__init__()
        # self.image_dir = image_dir + ('train/' if split == 'train' else 'test/')
        self.image_dir = image_dir + split + ('/' if len(split) else '')
        self.transform = transforms.Compose([
                        transforms.Resize(image_size),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        # self.norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_list = []
        
        with open(label2index_file, 'r') as f:
            self.label2index = json.load(f)

        self.cat_list = sorted(os.listdir(self.image_dir))

        for cat in self.cat_list:
            name_list = sorted(os.listdir(self.image_dir + cat))
            self.image_list += [self.image_dir + cat + '/' + image_name for image_name in name_list]

        print('Total %d Data.' % len(self.image_list))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = self.image_list[index]
        label = image_path.split('/')[-2] # label name

        index = self.label2index[label]
        index = torch.LongTensor([index]).squeeze()

        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, index

def get_train_loader(exp_name):
    if 'car' in exp_name:
        dataset_dir = '/export/d3/user/dataset/stanford-car-dataset/car_data/car_data/'
        train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(15),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.ImageFolder(root=dataset_dir+'train', transform=train_tfms)
    elif 'animal' in exp_name:
        dataset_dir = '/export/d3/user/dataset/animal-faces/afhq/'
        train_tfms = transforms.Compose([transforms.Resize((400, 400)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomRotation(15),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = torchvision.datasets.ImageFolder(root=dataset_dir+'train', transform=train_tfms)
    elif 'face' in exp_name:
        dataset = CelebADataset(split='train')
    else:
        dataset = ImageNetDataset(split='train-20')
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    return train_loader