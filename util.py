import os
import numpy as np
import pandas as pd
import shutil

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
def save_submission(result_dir, prediction, epoch, batch):
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    suffix = 'epoch{}_batch{}_date{}'.format(epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    pred = flatten(prediction)
    submission = pd.read_csv('./data/submission.csv')
    submission['digit'] = fn_tonumpy(torch.LongTensor(pred))
    submission.to_csv('./result/submission_{}.csv'.format(suffix), index=False)

## save model
def save_model(ckpt_dir, net, optim, num_epoch, epoch, batch, best_loss, save_argument):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    suffix = 'model_epoch{}_batch{}_date{}'.format(epoch, batch, datetime.now().strftime("%m.%d-%H:%M"))

    filename = os.path.join(ckpt_dir, 'checkpoint')
    best_filename = os.path.join(ckpt_dir, 'best_checkpoint')
    final_filename = os.path.join(ckpt_dir, 'final_' + suffix)

    # save model every epoch
    torch.save({'epoch' : epoch, 'net': net.state_dict(), 'optim': optim.state_dict(),
                'best_loss' : best_loss}, filename)

    # leave only best model
    if save_argument:
        shutil.copyfile(filename, best_filename)

    if num_epoch == epoch:
        shutil.copyfile(best_filename, final_filename)

## 네트워크 불러오기
def load(ckpt_dir, net, optim):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return net, optim, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]))

    net.load_state_dict(dict_model['net'])
    optim.load_state_dict(dict_model['optim'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('_batch')[0])

    return net, optim, epoch