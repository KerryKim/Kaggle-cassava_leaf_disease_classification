'''
!python3 './main.py' \
--mode "train" --train_continue "off" \
--seed 719 --img_x 512 --img_y 512 \
--lr 1e-4 --num_fold 10 --num_epoch 10 --batch_size 8 \
--cutmix False --fmix False \
--label_smooth False --swa False \
--data_dir "./data/train_images" \
--ckpt_dir "./checkpoint" \
--result_dir "./result" \
--network "tf_efficientnet_b4_ns"


if mode == 'train':
    %matplotlib inline
    %config InlineBackend.figure_format = 'retina'

    plt.plot(train_loss, label='Training loss')
    plt.plot(val_loss, label='Validation loss')
    plt.legend(frameon=False)

    plt.figure() # 하나의 윈도우를 나타냄, 생략가능
    plt.plot(train_acc, label='Training accuracy')
    plt.plot(val_acc, label='Validation accuracy')


    plt.legend(frameon=False)
'''

## 라이브러리 추가하기
import os
import numpy as np
import random

import torch

import argparse
from train import *

## Parser 생성하기
parser = argparse.ArgumentParser(description="Cassava Leaf Disease Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="on", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--seed", default=2021, type=int, dest="seed")
parser.add_argument("--img_x", default=224, type=int, dest="img_x")
parser.add_argument("--img_y", default=224, type=int, dest="img_y")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--num_fold", default=10, type=int, dest="num_fold")
parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")
parser.add_argument("--batch_size", default=4, type=int, dest="batch_size")

parser.add_argument("--cutmix", default=False, choices=[True, False], type=bool, dest="cutmix")
parser.add_argument("--fmix", default=False, choices=[True, False], type=bool, dest="fmix")

parser.add_argument("--label_smooth", default=False, choices=[True, False], type=bool, dest="label_smooth")
parser.add_argument("--swa", default=False, choices=[True, False], type=bool, dest="swa")

parser.add_argument("--data_dir", default="./data/train_images", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--network", default="tf_efficientnet_b4_ns", type=str, dest="network")

args = parser.parse_args()

##
if __name__ == "__main__":
    # random seed
    def seed_everything(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

    seed = args.seed
    seed_everything(seed)

    if args.mode == "train":
        train(args)
    # elif args.mode == "test":
    #     test(args)

# tensorboard --logdir ./log/train
