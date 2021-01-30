## 라이브러리 추가하기
import os
import numpy as np
import random

import torch

import argparse
from train import *

import warnings
warnings.filterwarnings('ignore')

## Parser 생성하기
parser = argparse.ArgumentParser(description="Cassava Leaf Disease Classification",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--mode", default="train", choices=["train", "test"], type=str, dest="mode")

parser.add_argument("--seed", default=719, type=int, dest="seed")
parser.add_argument("--img_x", default=512, type=int, dest="img_x") # 384, 512
parser.add_argument("--img_y", default=512, type=int, dest="img_y")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--num_fold", default=5, type=int, dest="num_fold")
parser.add_argument("--num_epoch", default=20, type=int, dest="num_epoch")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")

parser.add_argument("--label_smooth", default=True, choices=[True, False], type=bool, dest="label_smooth")
parser.add_argument("--swa", default=False, choices=[True, False], type=bool, dest="swa")

parser.add_argument("--data_dir", default="./data/train_images_c", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--model", default="./checkpoint/model", type=str, dest="model")

parser.add_argument("--network", default="tf_efficientnet_b4_ns", type=str, dest="network")
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")

args, unknown = parser.parse_known_args()

##
if __name__ == "__main__":
    seed = args.seed
    seed_everything(seed)

    if args.mode == "train":
        train(args)

# tensorboard --logdir ./log/train
