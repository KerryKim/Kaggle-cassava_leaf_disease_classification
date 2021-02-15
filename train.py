##
import os
import sys
import numpy as np
import pandas as pd

sys.path.append('/home/kerrykim/jupyter_notebook/010.cldc/etc/Ranger/pytorch_ranger')

# pytorch
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from pytorch_ranger import ranger

from torch.cuda.amp import autocast, GradScaler
from torchcontrib.optim import SWA

# sci-kit learn
from sklearn.model_selection import StratifiedKFold

# augmentation
import albumentations as A
from albumentations.pytorch import ToTensorV2

# etc
from tqdm import tqdm

from model import *
from loss import *
from dataset import *
from util import *
from main import *

LOGGER = init_logger()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



##
def transform_train(img_x, img_y):
    return A.Compose([A.Transpose(p=0.5),
                      A.HorizontalFlip(p=0.5),
                      A.VerticalFlip(p=0.5),
                      A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.5),
                      A.HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
                      A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
                      A.RandomResizedCrop(img_x, img_y),
                      A.CoarseDropout(max_holes=12, max_height=int(0.11*CFG.img_y), max_width=int(0.11*CFG.img_x),
                                      min_holes=1, min_height=int(0.03*CFG.img_y), min_width=int(0.03*CFG.img_x),
                                      always_apply=False, p=0.5),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                      ToTensorV2(p=1.0)], p=1.)


def transform_val(img_x, img_y):
    return A.Compose([A.CenterCrop(img_x, img_y, p=1.),
                      A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
                      ToTensorV2(p=1.0)], p=1.)



##
def train_one_epoch(loader_train, net, fn_loss, optim, epoch, scaler, device):
    # fn_help
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    net.train()
    start = end = time.time()

    for batch, data in enumerate(loader_train, 1):
        # measure data loading time
        data_time.update(time.time() - end)

        # forward pass
        label = data['label'].to(device)
        input = data['input'].to(device)

        with autocast():
            output = net(input)
            loss = fn_loss(output, label)

        # record loss
        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # Normalize for Gradient accumulation
        if CFG.gradient_accumulation_step > 1:
            loss = loss / CFG.gradient_accumulation_step

        # backward pass
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=CFG.max_grad_norm)    # clipping grad

        # Gradient accumulation (After 2 batch, update gradient)
        if (batch % CFG.gradient_accumulation_step == 0) or (batch == len(loader_train)):
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == len(loader_train):
            print('Epoch {0}: [{1}/{2}] '
                # 'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                # 'LR: {lr:.6f}  '
                .format(
                epoch, batch, len(loader_train), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(batch) / len(loader_train)),
                grad_norm=grad_norm,
                # lr=scheduler.get_lr()[0]
            ))

    return losses.avg



##
def val_one_epoch(loader_val, net, fn_loss, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    net.eval()
    acc = 0.0
    start = end = time.time()

    for batch, data in enumerate(loader_val, 1):
        data_time.update(time.time() - end)

        label = data['label'].to(device)
        input = data['input'].to(device)

        with torch.no_grad():
            output = net(input)
            loss = fn_loss(output, label)

        batch_size = label.size(0)
        losses.update(loss.item(), batch_size)

        # record accuracy
        output = F.softmax(output).argmax(axis=1)
        acc += (output==label).sum().cpu().numpy()    # n of pred==label

        batch_time.update(time.time() - end)
        end = time.time()

        if batch % CFG.print_freq == 0 or batch == (len(loader_val)):
            print('EVAL: [{0}/{1}] '
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f} (avg {loss.avg:.4f}) '
                .format(
                batch, len(loader_val), batch_time=batch_time,
                data_time=data_time, loss=losses,
                remain=timeSince(start, float(batch) / len(loader_val)),
            ))

    val_acc = acc / len(loader_val) / CFG.batch_size

    return losses.avg, val_acc



##
def train(df):
    df_20 = df.loc[df.source == 2020]
    df_19 = df.loc[df.source == 2019]

    skf = StratifiedKFold(n_splits=CFG.num_fold, shuffle=True, random_state=CFG.seed)
    # skf.get_n_splits(np.arange(train_df.shape[0]), train_df['label'])

    KFOLD = [(idxT, idxV) for i, (idxT, idxV) in enumerate(skf.split(np.arange(df_20.shape[0]), df_20['label']))]
    KFOLD_19 = [np.concatenate((idxT, idxV)) for i, (idxT, idxV) in
                  enumerate(skf.split(np.arange(df_19.shape[0]), df_19['label']))]

    for i in range(CFG.num_fold):
        (idxT, idxV) = KFOLD[i]
        # KFOLD 20 + 19
        KFOLD[i] = (np.concatenate((idxT, df_19.iloc[KFOLD_19[i]].index)), idxV)
        (idxT, idxV) = KFOLD[i]

        # When train, 20,19 data all used. / When val, 20 data only used.
        print(np.bincount(df['label'].iloc[idxT]), np.bincount(df_20['label'].iloc[idxV]))

    for fold, (trn_idx, val_idx) in enumerate(KFOLD, 1):
        LOGGER.info(f"Training starts ... KFOLD: {fold}/{CFG.num_fold}")

        train = df.loc[trn_idx, :].reset_index(drop=True)
        val = df.loc[val_idx, :].reset_index(drop=True)

        dataset_train = Dataset(df=train, data_dir=CFG.data_dir, transform=transform_train(CFG.img_x, CFG.img_y))
        loader_train = DataLoader(dataset_train, batch_size=CFG.batch_size, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)

        dataset_val = Dataset(df=val, data_dir=CFG.data_dir, transform=transform_val(CFG.img_x, CFG.img_y))
        loader_val = DataLoader(dataset_val, batch_size=CFG.batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

        net = CassvaImgClassifier(CFG.model, df.label.nunique(), pretrained=True).to(device)
        # fn_loss = nn.CrossEntropyLoss().to(device)
        fn_loss = TaylorSmoothedLoss().to(device)
        optim = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=1e-6, amsgrad=False) # 1e-6
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=7, T_mult=1, eta_min=8e-7, last_epoch=-1)
        scaler = GradScaler()


        # default value
        st_epoch = 0
        best_score = 0.
        early_stop_patience = 0


        for epoch in range(st_epoch + 1, CFG.num_epoch + 1):
            start_time = time.time()

            # train
            avg_train_loss = train_one_epoch(loader_train, net, fn_loss, optim, epoch, scaler, device)

            # val
            avg_val_loss, score = val_one_epoch(loader_val, net, fn_loss, device)

            scheduler.step()

            # scoring
            elapsed = time.time() - start_time

            LOGGER.info(
                f'Epoch {epoch} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
            LOGGER.info(f'Epoch {epoch} - Accuracy: {score}')

            # early-stopping
            early_stop_patience += 1

            # if loss/acc is improve, esp starts 0
            if score > best_score:
                early_stop_patience = 0
            if early_stop_patience > 3:
                print("Training early stopped. Loss/Acc was not improved")
                LOGGER.info(f'Epoch {epoch} - Save Best Score: {best_score:.4f} Model')
                epoch = epoch - 3
                save_model(ckpt_dir=CFG.ckpt_dir, net=net, fold=fold, num_epoch=epoch, epoch=epoch,
                           batch=CFG.batch_size, save_argument=save_argument)
                break

            # save best model
            save_argument = score > best_score  # result is True
            best_score = max(score, best_score)

            LOGGER.info(f'Epoch {epoch} - Save Best Score: {best_score:.4f} Model')

            save_model(ckpt_dir=CFG.ckpt_dir, net=net, fold=fold, num_epoch=CFG.num_epoch, epoch=epoch,
                       batch=CFG.batch_size, save_argument=save_argument)


