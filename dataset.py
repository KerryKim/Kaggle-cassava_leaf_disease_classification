import os
import sys
import numpy as np
import cv2

import torch
from util import *

# Fmix-master folder rigit click -> mark directory as -> resources
sys.path.append('/home/kerrykim/jupyter_notebook/010.cldc/FMix-master')
from fmix import *

##
class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, img_x, img_y, transform=None, cutmix=False, fmix=False):
        self.df = df
        self.data_dir = data_dir
        self.img_x = img_x
        self.img_y = img_y
        self.transform = transform
        self.cutmix = cutmix
        self.fmix = fmix
        lst_label = list(df['label'])
        lst_input = list(os.path.join(data_dir, x) for x in df.image_id.values)
        self.lst_label = lst_label
        self.lst_input = lst_input


    def __len__(self):
        return len(self.lst_label)


    def __getitem__(self, index):
        label = self.lst_label[index]
        # input = cv2.imread(self.lst_input[index], cv2.IMREAD_COLOR)
        # input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # result of input shape is y,x,c
        input = cv2.imread(self.lst_input[index])
        input = input[:, :, ::-1]

        if self.transform:
            input = self.transform(image=input)['image']

        if self.cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            # with torch.no_grad():
            cmix_idx = np.random.choice(self.df.index, size=1)[0]
            cmix_img = cv2.imread(os.path.join(self.data_dir, self.df.iloc[cmix_idx]['image_id']))
            cmix_img = cv2.cvtColor(cmix_img, cv2.COLOR_BGR2RGB)
            if self.transform:
                cmix_img = self.transform(image=cmix_img)['image']
            lam = np.clip(np.random.beta(1, 1), 0.3, 0.4)
            bbx1, bby1, bbx2, bby2 = rand_bbox(self.img_x, self.img_y, lam)
            input[bbx1:bbx2, bby1:bby2, :] = cmix_img[bbx1:bbx2, bby1:bby2, :]
            rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.img_x * self.img_y))
            label = rate * label + (1. - rate) * self.df['label'][cmix_idx]


        if self.fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            # with torch.no_grad():
            lam = np.clip(np.random.beta(1., 1.), 0.6, 0.7)
            # Make mask, get mean / std
            mask = make_low_freq_image(3, (self.img_x, self.img_y))
            mask = binarise_mask(mask, lam, (self.img_x, self.img_y), max_soft=True)
            mask = mask.reshape(self.img_x, self.img_y, 1)
            fmix_idx = np.random.choice(self.df.index, size=1)[0]
            fmix_img = cv2.imread(os.path.join(self.data_dir, self.df.iloc[fmix_idx]['image_id']))
            fmix_img = cv2.cvtColor(fmix_img, cv2.COLOR_BGR2RGB)
            if self.transform:
                fmix_img = self.transform(image=fmix_img)['image']
            mask_torch = torch.from_numpy(mask)
            # mix image
            input = mask_torch * input + (1. - mask_torch) * fmix_img
            input = input.to('cpu').detach().numpy()
            # mix target
            rate = mask.sum() / self.img_x / self.img_y
            label = rate * label + (1. - rate) * self.df['label'][fmix_idx]

        data = {'input' : input, 'label' : label}

        return data

##
class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, img_x, img_y, transform=None):
        self.data_dir = data_dir
        self.img_x = img_x
        self.img_y = img_y
        self.transform = transform

        lst_input = ['./data/test_images/2216849948.jpg']
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_input)

    def __getitem__(self, index):
        input = cv2.imread(self.lst_input[index], cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # result of input shape is y,x,c

        if self.transform:
            input = self.transform(image=input)['image']

        data = {'input': input}

        return data


##

def rand_bbox(img_x, img_y, lam):    # size = image size, lam = beta distribution
    W = img_x
    H = img_y
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2