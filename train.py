##
import os
import sys
import numpy as np
import pandas as pd
import random

# pytorch
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp.autocast_mode import autocast
from torch.cuda.amp import GradScaler
from torchcontrib.optim import SWA

# sci-kit learn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

# from Lookahead import Lookahead
# from RAdam import RAdam
# from bi_tempered_loss import *

from tqdm import tqdm

from model import *
from loss import *
from dataset import *
from util import *

##
# random seed
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

##
def transform_train(img_x, img_y):
    return A.Compose([A.RandomResizedCrop(img_x, img_y),
                      A.Resize(img_x, img_y),
                      A.Transpose(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.ShiftScaleRotate(p=0.5),
                      # A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                      # A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                      A.CoarseDropout(p=0.5), ToTensorV2(p=1.0)], p=1.)


def transform_val(img_x, img_y):
    return A.Compose([A.CenterCrop(img_x, img_y, p=1.),
                      A.Resize(img_x, img_y),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),ToTensorV2(p=1.0)], p=1.)

##
def dataloader_train(df, data_dir, trn_idx, val_idx, img_x, img_y, batch_size):
    train = df.loc[trn_idx, :].reset_index(drop=True)
    val = df.loc[val_idx, :].reset_index(drop=True)

    dataset_train = Dataset(df=train, data_dir=data_dir,
                            img_x=img_x, img_y=img_y, transform=transform_train(img_x, img_y))
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = Dataset(df=val, data_dir=data_dir,
                          img_x=img_x, img_y=img_y, transform=transform_val(img_x, img_y))
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

    return loader_train, loader_val, num_batch_train, num_batch_val


##
def train(args):
    # hyperparameters
    seed = args.seed
    img_x = args.img_x
    img_y = args.img_y

    lr = args.lr
    num_fold = args.num_fold
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    label_smooth = args.label_smooth
    swa = args.swa

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    network = args.network

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##
    df = pd.read_csv('./data/train_c_label5.csv')
    Kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(Kfold.split(np.arange(df.shape[0]), df.label.values), 1):
        #if fold > 1:
            #break

        print("Training start ... ")
        print("DATE: {} | ".format(datetime.now().strftime("%m.%d-%H:%M")), "KFOLD: {}/{}".format(fold, num_fold))

        loader_train, loader_val, num_batch_train, num_batch_val = \
            dataloader_train(df, data_dir, trn_idx, val_idx, img_x, img_y, batch_size)

        net = CassvaImgClassifier(network, df.label.nunique(), pretrained=True).to(device)
        fn_loss = nn.CrossEntropyLoss().to(device) if not label_smooth else Label_Smooth_CrossEntropyLoss
        optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1)
        scaler = GradScaler()

        if swa:
            SWA(optim, swa_start=10, swa_freq=2, swa_lr=0.0005)

        ##
        st_epoch = 0
        best_loss = 1e20
        best_acc = 0
        early_stop_patience = 0

        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            loss_arr = []  # loss array for one batch

            # loss/accuracy for one epoch
            loss_epoch_train = 0
            acc_epoch_train = 0

            pbar_train = tqdm(enumerate(loader_train), total=len(loader_train))

            for batch, data in pbar_train:
                # forward pass
                label = data['label'].to(device).long()
                input = data['input'].to(device).float()

                with autocast():
                    output = net(input)
                    loss = fn_loss(output, label)
                    scaler.scale(loss).backward()

                # Gradient accumulation (After 2 batch, update gradient)
                if ((batch + 1) % 2 == 0) or ((batch + 1) == len(loader_train)):
                    # backward pass
                    scaler.step(optim)
                    scaler.update()
                    optim.zero_grad()

                loss_arr += [loss.item()]
                loss_batch_train = np.mean(loss_arr)
                loss_epoch_train += loss_batch_train

                pbar_train.set_description("loss_batch_trn : %4f" % loss_batch_train)

                output = torch.argmax(output, dim=1).cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                acc_batch_train = accuracy_score(label, output)
                acc_epoch_train += acc_batch_train

            train_loss.append(loss_epoch_train / num_batch_train)
            train_acc.append(acc_epoch_train / num_batch_train)

            with torch.no_grad():
                net.eval()
                loss_arr = []

                loss_epoch_val = 0
                acc_epoch_val = 0

                pbar_val = tqdm(enumerate(loader_val), total=len(loader_val))

                for batch, data in pbar_val:
                    label = data['label'].to(device).long()
                    input = data['input'].to(device).float()

                    output = net(input)

                    loss = fn_loss(output, label)

                    loss_arr += [loss.item()]
                    loss_batch_val = np.mean(loss_arr)
                    loss_epoch_val += loss_batch_val

                    pbar_val.set_description("loss_batch_val : %4f" % loss_batch_val)

                    output = torch.argmax(output, dim=1).cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    acc_batch_val = accuracy_score(label, output)
                    acc_epoch_val += acc_batch_val

                val_loss.append(loss_epoch_val / num_batch_val)
                val_acc.append(acc_epoch_val / num_batch_val)

            scheduler.step(val_loss[-1])

            print("DATE: {} | ".format(datetime.now().strftime("%m.%d-%H:%M")), "EPOCH: {}/{} | ".format(epoch, num_epoch),
                  "TRAIN_LOSS: {:4f} | ".format(train_loss[-1]),  "TRAIN_ACC: {:4f} | ".format(train_acc[-1]),
                  "VAL_LOSS: {:4f} | ".format(val_loss[-1]), "VAL_ACC: {:4f} | ".format(val_acc[-1]))

            # early-stopping
            early_stop_patience += 1
            # if loss/acc is improve, esp starts 0
            if best_loss > val_loss[-1] and best_acc <= val_acc[-1]:
                early_stop_patience = 0
            if early_stop_patience > 3:
                print("Training early stopped. Loss/Acc was not improved")
                break

            # save best model
            save_argument = best_loss > val_loss[-1]   # result is True
            best_loss = min(val_loss[-1], best_loss)
            best_acc = max(val_acc[-1], best_acc)

            save_model(ckpt_dir=ckpt_dir, net=net, optim=optim, fold=fold, num_epoch=num_epoch, epoch=epoch,
                       batch=batch_size, best_loss=best_loss, save_argument=save_argument)

        if swa:
            try:
                optim.swap_swa_sgd()
            except:
                pass

