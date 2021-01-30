import os
import cv2
import numpy as np

import torch

# Fmix-master folder rigit click -> mark directory as -> resources
import sys
sys.path.append('/home/kerrykim/jupyter_notebook/010.cldc/FMix-master')
from fmix import *

##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, img_x, img_y, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.img_x = img_x
        self.img_y = img_y
        self.transform = transform

        lst_label = list(df['label'])
        lst_input = list(x for x in df.image_id.values)

        self.lst_label = lst_label
        self.lst_input = lst_input


    def __len__(self):
        return len(self.lst_label)


    def __getitem__(self, index):
        label = self.lst_label[index]
        input = cv2.imread(os.path.join(self.data_dir, self.lst_input[index]), cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # result of input shape is y,x,c

        if self.transform:
            input = self.transform(image=input)['image']

        data = {'input' : input, 'label' : label}

        return data

