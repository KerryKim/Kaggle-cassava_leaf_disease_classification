import os
import shutil
import numpy as np
import pandas as pd

import torch

from datetime import datetime

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
    submission.to_csv('./submission.csv', index=False)

## save model
def save_model(ckpt_dir, net, optim, fold, num_epoch, epoch, batch, best_loss, save_argument):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'f{}_ep{}_bt{}_date{}'.format(fold, epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    filename = os.path.join(ckpt_dir, 'checkpoint_0')
    best_filename = os.path.join(ckpt_dir, 'best_checkpoint_0')
    final_filename = os.path.join(ckpt_dir, 'final_' + suffix)

    # save model every epoch
    # If you want to save model every epoch, change filename to suffix
    torch.save({'epoch' : epoch, 'net': net.state_dict(), 'optim': optim.state_dict(),
                'best_loss' : best_loss}, filename)

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
