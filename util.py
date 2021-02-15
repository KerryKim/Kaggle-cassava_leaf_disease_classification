import os
import math
import time
import random

import shutil
import numpy as np
import pandas as pd

import torch

from sklearn.metrics import accuracy_score

from datetime import datetime


##
OUTPUT_DIR = './'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)



##
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0    # val is value
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))



##
def get_score(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def init_logger(log_file=OUTPUT_DIR+'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger



##
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



## from tensor to numpy function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()



## 리스트 펼치기 함수
def flatten(lst):
    result = []
    for item in lst:
        result.extend(item)
    return result



## 제출파일 저장하기
def save_submission(data_dir, pred):
    submission = pd.DataFrame()
    submission['image_id'] = list(os.listdir(data_dir))
    submission['label'] = pred
    submission.to_csv('./result/submission.csv', index=False)



## save model
def save_model(ckpt_dir, net, fold, num_epoch, epoch, batch, save_argument):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'f{}_ep{}_bt{}_date{}.pth'.format(fold, epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    filename = os.path.join(ckpt_dir, 'checkpoint_0.pth')
    best_filename = os.path.join(ckpt_dir, 'best_checkpoint_0.pth')
    final_filename = os.path.join(ckpt_dir, 'final_' + suffix)

    # save model every epoch
    # If you want to save model every epoch, change filename to suffix
    torch.save(net.state_dict(), filename)

    # leave only best model
    if save_argument:
        shutil.copyfile(filename, best_filename)

    if num_epoch == epoch:
        shutil.copyfile(best_filename, final_filename)
        os.remove(filename)
        os.remove(best_filename)



## 네트워크 불러오기
def load(model, net, optim):
    dict_model = torch.load(model)
    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    return net, optim
