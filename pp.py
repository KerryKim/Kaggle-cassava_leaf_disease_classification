import sys


package_paths = [
    './efficientnet_pytorch-0.7.0', #'../input/efficientnet-pytorch-07/efficientnet_pytorch-0.7.0'
    './FMix-master']

for pth in package_paths:
    sys.path.append(pth)

# Fmix-master folder rigit click -> mark directory as -> resources
sys.path.append('/home/kerrykim/jupyter_notebook/010.cldc/FMix-master/')

print(sys.path)

from fmix import *