##
import os
import sys
import numpy as np
import pandas as pd

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


from model import *
from loss import *
from dataset import *
from util import *

##
def transform_train(img_x, img_y):
    return A.Compose([A.RandomResizedCrop(img_x, img_y),
                      A.Transpose(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.ShiftScaleRotate(p=0.5),
                      A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                      A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                      A.CoarseDropout(p=0.5), A.Cutout(p=0.5),ToTensorV2(p=1.0)], p=1.)


def transform_val(img_x, img_y):
    return A.Compose([A.CenterCrop(img_x, img_y, p=1.),
                      A.Resize(img_x, img_y),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),ToTensorV2(p=1.0)], p=1.)


def transform_test(img_x, img_y):
    return A.Compose([
            A.RandomResizedCrop(img_x, img_y),
            A.Transpose(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0)], p=1.)

##
def dataloader_train(df, data_dir, trn_idx, val_idx, img_x, img_y, batch_size, cutmix, fmix):
    train = df.loc[trn_idx, :].reset_index(drop=True)
    val = df.loc[val_idx, :].reset_index(drop=True)

    dataset_train = TrainDataset(df=train, data_dir=data_dir, img_x=img_x, img_y=img_y, transform=transform_train(img_x, img_y), cutmix=cutmix, fmix=fmix)
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8)

    dataset_val = TrainDataset(df=val, data_dir=data_dir, img_x=img_x, img_y=img_y, transform=transform_val(img_x, img_y), cutmix=False, fmix=False)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, num_workers=8)

    num_data_train = len(dataset_train)
    num_data_val = len(dataset_val)

    num_batch_train = np.ceil(num_data_train / batch_size)
    num_batch_val = np.ceil(num_data_val / batch_size)

    return loader_train, loader_val, num_batch_train, num_batch_val

##
def dataloader_test(data_dir, img_x, img_y, batch_size):
    dataset_test = TestDataset(data_dir=data_dir, img_x=img_x, img_y=img_y, transform=transform_test(img_x, img_y))
    loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=8)

    num_data_test = len(dataset_test)
    num_batch_test = np.ceil(num_data_test / batch_size)

    return loader_test, num_batch_test

##
def train(args):
    # hyperparameters
    train_continue = args.train_continue
    seed = args.seed

    img_x = args.img_x
    img_y = args.img_y

    lr = args.lr
    num_fold = args.num_fold
    num_epoch = args.num_epoch
    batch_size = args.batch_size

    cutmix = args.cutmix
    fmix = args.fmix

    label_smooth = args.label_smooth
    swa = args.swa

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir

    network = args.network

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ##
    df = pd.read_csv('./data/train.csv')
    Kfold = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)

    for fold, (trn_idx, val_idx) in enumerate(Kfold.split(np.arange(df.shape[0]), df.label.values)):
        if fold > 0:
            break

        print("Training start ... ")
        print("DATE: {} | ".format(datetime.now().strftime("%m.%d-%H:%M")), "KFOLD: {}/{}".format(fold, num_fold))

        loader_train, loader_val, num_batch_train, num_batch_val = \
            dataloader_train(df, data_dir, trn_idx, val_idx, img_x, img_y, batch_size, cutmix, fmix)

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

        if train_continue == "on":
            net, optim, st_epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

        train_loss, val_loss, train_acc, val_acc = [], [], [], []

        for epoch in range(st_epoch + 1, num_epoch + 1):
            net.train()
            loss_arr = []  # loss array for one batch

            # loss/accuracy for one epoch
            loss_epoch_train = 0
            acc_epoch_train = 0

            for batch, data in enumerate(loader_train, 1):
                # forward pass
                label = data['label'].to(device)
                input = data['input'].to(device)

                with autocast():
                    output = net(input)
                    loss = fn_loss(output, label)
                    optim.zero_grad()

                # backward pass
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()

                # loss.backward()
                # optim.step()

                loss_arr += [loss.item()]
                loss_batch_train = np.mean(loss_arr)
                loss_epoch_train += loss_batch_train

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

                for batch, data in enumerate(loader_val, 1):
                    label = data['label'].to(device)
                    input = data['input'].to(device)

                    output = net(input)

                    loss = fn_loss(output, label)

                    loss_arr += [loss.item()]
                    loss_batch_val = np.mean(loss_arr)
                    loss_epoch_val += loss_batch_val

                    output = torch.argmax(output, dim=1).cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    acc_batch_val = accuracy_score(label, output)
                    acc_epoch_val += acc_batch_val

                scheduler.step(loss_batch_val)

                val_loss.append(loss_epoch_val / num_batch_val)
                val_acc.append(acc_epoch_val / num_batch_val)

            print("DATE: {} | ".format(datetime.now().strftime("%m.%d-%H:%M")), "EPOCH: {}/{} | ".format(epoch, num_epoch),
                  "TRAIN_LOSS: {:4f} | ".format(train_loss[-1]),  "TRAIN_ACC: {:4f} | ".format(train_acc[-1]),
                  "VAL_LOSS: {:4f} | ".format(val_loss[-1]), "VAL_ACC: {:4f} | ".format(val_acc[-1]))

            # early-stopping
            early_stop_patience += 1
            # if loss/acc is improve, esp starts 0
            if best_loss > val_loss[-1] and best_acc <= val_acc[-1]:
                early_stop_patience = 0
            if early_stop_patience > 10:
                print("Training early stopped. Loss/Acc was not improved")
                break

            # save best model
            save_argument = best_loss > val_loss[-1]   # result is True
            best_loss = min(val_loss[-1], best_loss)
            best_acc = max(val_acc[-1], best_acc)

            save_model(ckpt_dir=ckpt_dir, net=net, optim=optim, num_epoch=num_epoch, epoch=epoch,
                       batch=batch_size, best_loss=best_loss, save_argument=save_argument)

        if swa:
            try:
                optim.swap_swa_sgd()
            except:
                pass


##
def test(args):
    # hyperparameters
    img_x = args.img_x
    img_y = args.img_y

    lr = args.lr
    batch_size = args.batch_size

    num_epoch = args.num_epoch

    data_dir = args.data_dir
    ckpt_dir = args.ckpt_dir
    result_dir = args.result_dir

    network = args.network

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##
    print("Test start ... ")
    df = pd.read_csv('./data/train.csv')

    loader_test, num_batch_test = dataloader_test(data_dir=data_dir, img_x=img_x, img_y=img_y, batch_size=batch_size)
    net = CassvaImgClassifier(network, df.label.nunique(), pretrained=True).to(device)
    optim = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-6)
    net, optim, epoch = load(ckpt_dir=ckpt_dir, net=net, optim=optim)

    #used_epoch = [6, 7, 8, 9]  if test use rnd_epochs, it can have weights for each iteration.


    with torch.no_grad():
        net.eval()
        st_iter = 0
        tta = 5
        pred = []

        for iter in range(st_iter + 1, tta + 1):
            for batch, data in enumerate(loader_test, 1):
                # forward pass
                input = data['input'].to(device)

                output = net(input)

                output = torch.softmax(output, dim=1).cpu().detach().numpy()
                pred.append(output)

                print("TTA ITERATION: {}/{} | ".format(iter, tta), "BATCH: %04d / %04d" % (batch, num_batch_test))

        print(pred)
        pred = (np.mean(pred, axis=0)).argmax(axis=1)

        # submission
        if batch % num_batch_test == 0:
            save_submission(result_dir=result_dir, prediction=pred, epoch=num_epoch, batch=batch_size)

