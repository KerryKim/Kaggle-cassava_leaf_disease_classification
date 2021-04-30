import os
import cv2

import torch


##
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, transform=None):
        self.df = df
        self.data_dir = data_dir
        self.transform = transform

        self.arr_label = df['label'].values
        self.arr_input = df['image_id'].values

    def __len__(self):
        return len(self.arr_label)

    def __getitem__(self, index):
        label = self.arr_label[index] # label = torch.tensor(self.arr_label[index]).long()
        input = cv2.imread(os.path.join(self.data_dir, self.arr_input[index]), cv2.IMREAD_COLOR)
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)  # result of input shape is y,x,c

        if self.transform:
            input = self.transform(image=input)['image']

        data = {'input' : input, 'label' : label}

        return data

