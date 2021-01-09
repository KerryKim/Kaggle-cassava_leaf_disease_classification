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

parser.add_argument("--mode", default="test", choices=["train", "test"], type=str, dest="mode")
parser.add_argument("--train_continue", default="off", choices=["on", "off"], type=str, dest="train_continue")

parser.add_argument("--seed", default=719, type=int, dest="seed")
parser.add_argument("--img_x", default=512, type=int, dest="img_x")
parser.add_argument("--img_y", default=512, type=int, dest="img_y")

parser.add_argument("--lr", default=1e-4, type=float, dest="lr")
parser.add_argument("--num_fold", default=10, type=int, dest="num_fold")
parser.add_argument("--num_epoch", default=10, type=int, dest="num_epoch")
parser.add_argument("--batch_size", default=8, type=int, dest="batch_size")

parser.add_argument("--cutmix", default=False, choices=[True, False], type=bool, dest="cutmix")
parser.add_argument("--fmix", default=False, choices=[True, False], type=bool, dest="fmix")

parser.add_argument("--label_smooth", default=False, choices=[True, False], type=bool, dest="label_smooth")
parser.add_argument("--swa", default=False, choices=[True, False], type=bool, dest="swa")

parser.add_argument("--data_dir", default="./etc/sample_data", type=str, dest="data_dir")
parser.add_argument("--ckpt_dir", default="./checkpoint", type=str, dest="ckpt_dir")
parser.add_argument("--result_dir", default="./result", type=str, dest="result_dir")

parser.add_argument("--network", default="tf_efficientnet_b4_ns", type=str, dest="network")

args, unknown = parser.parse_known_args()

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
    elif args.mode == "test":
        test(args)

# tensorboard --logdir ./log/train
