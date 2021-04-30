## 라이브러리 추가하기
from train import *
from util import *

import warnings
warnings.filterwarnings('ignore')

## Configurations
class CFG:
    mode = 'train'
    seed = 42
    print_freq = 100

    N_CLASS = 5

    img_x = 512
    img_y = 512

    num_fold = 5
    num_epoch = 10
    batch_size = 8

    gradient_accumulation_step = 2
    max_grad_norm = 1000

    criterion_name = "TaylorSmoothedLoss"

    LABEL_SMOOTH = 0.20

    T1 = 0.2
    T2 = 1.1

    apex = True
    swa = False

    data_dir = './data/train_images_m'
    info_dir = './data/train_m.csv'
    ckpt_dir = './checkpoint'
    result_dir = './result'
    net_dir = './checkpoint/model'

    model = 'tf_efficientnet_b4_ns'


##
if __name__ == "__main__":
    seed_everything(CFG.seed)
    df = pd.read_csv(CFG.info_dir)

    if CFG.mode == 'train':
        torch.set_default_tensor_type('torch.FloatTensor')
        train(df)
