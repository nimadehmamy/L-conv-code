"""LieConv Baseline experiments. 
requires: 
https://github.com/mfinzi/LieConv

Usage: 
$ python3 run_LieConv_cifar100.py --epochs 100 --nlay 2 --ker 256 --lr 3e-3 --bn 0 --rot 1 --scr 1

--rot: rotate images
--scr: scramble images (fixed shuffling of pixels in all images)
--nlay: number of layers 
--bn: batchnorm (0: no batchnorm; used in baseline experiments)

"""


import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from oil.utils.utils import LoaderTo, cosLr, islice
from oil.tuning.study import train_trial
from oil.datasetup.datasets import split_dataset
from oil.utils.parallel import try_multigpu_parallelize
from oil.model_trainers.classifier import Classifier
from functools import partial
from torch.optim import Adam
from oil.tuning.args import argupdated_config
import copy
import lie_conv.lieGroups as lieGroups
import lie_conv.lieConv as lieConv
from lie_conv.lieConv import ImgLieResnet
# from lie_conv.datasets import MnistRotDataset, RotMNIST
from oil.datasetup.datasets import EasyIMGDataset

from lie_conv.utils import Named, export, Expression, FixedNumpySeed, RandomZrotation, GaussianNoise
from lie_conv.utils import Named
import numpy as np
from PIL import Image
from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, \
    verify_str_arg
from torchvision.datasets.vision import VisionDataset
import torchvision

class RotCIFAR10(EasyIMGDataset,torchvision.datasets.CIFAR10):
#     """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 10
    def __init__(self,*args,dataseed=0,transform=None,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        N = len(self)
        with FixedNumpySeed(dataseed):
            angles = torch.rand(N)*2*np.pi
        with torch.no_grad():
            # R = torch.zeros(N,2,2)
            # R[:,0,0] = R[:,1,1] = angles.cos()
            # R[:,0,1] = R[:,1,0] = angles.sin()
            # R[:,1,0] *=-1
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            # self.data = self.data.unsqueeze(1).float()
            self.data = torch.as_tensor(self.data.transpose((0,3,1,2))).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid)
            normalize = transforms.Normalize((127.5,) ,(255,))
            self.data = normalize(self.data)
    def __getitem__(self,idx):
        return self.data[idx], int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation


class CIFAR100(EasyIMGDataset,torchvision.datasets.CIFAR100):
#     """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 100
    def __init__(self,*args,dataseed=0,transform=None,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        N = len(self)
        with torch.no_grad():
            self.data = torch.as_tensor(self.data.transpose((0,3,1,2))).float()
            normalize = transforms.Normalize((127.5,) ,(255,))
            self.data = normalize(self.data)
    def __getitem__(self,idx):
        return self.data[idx], int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation
    

class RotCIFAR100(EasyIMGDataset,torchvision.datasets.CIFAR100):
#     """ Unofficial RotMNIST dataset created on the fly by rotating MNIST"""
    means = (0.5,)
    stds = (0.25,)
    num_targets = 100
    def __init__(self,*args,dataseed=0,transform=None,**kwargs):
        super().__init__(*args,download=True,**kwargs)
        N = len(self)
        with FixedNumpySeed(dataseed):
            angles = torch.rand(N)*2*np.pi
        with torch.no_grad():
            # R = torch.zeros(N,2,2)
            # R[:,0,0] = R[:,1,1] = angles.cos()
            # R[:,0,1] = R[:,1,0] = angles.sin()
            # R[:,1,0] *=-1
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(N,2,3)
            affineMatrices[:,0,0] = angles.cos()
            affineMatrices[:,1,1] = angles.cos()
            affineMatrices[:,0,1] = angles.sin()
            affineMatrices[:,1,0] = -angles.sin()
            # affineMatrices[:,0,2] = -2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/w
            # affineMatrices[:,1,2] = 2*np.random.randint(-self.max_trans, self.max_trans+1, bs)/h
            # self.data = self.data.unsqueeze(1).float()
            self.data = torch.as_tensor(self.data.transpose((0,3,1,2))).float()
            flowgrid = F.affine_grid(affineMatrices, size = self.data.size())
            self.data = F.grid_sample(self.data, flowgrid)
            normalize = transforms.Normalize((127.5,) ,(255,))
            self.data = normalize(self.data)
    def __getitem__(self,idx):
        return self.data[idx], int(self.targets[idx])
    def default_aug_layers(self):
        return RandomRotateTranslate(0)# no translation
         
class RotScramCIFAR100(RotCIFAR100):
    """ Scrambled"""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        with torch.no_grad():
            idx = torch.randperm(self.data[0,0].nelement())
            self.data = self.data.view(*self.data.shape[:2], -1)[:,:,idx].view(self.data.size())

def makeTrainer(*, dataset=RotCIFAR100, network=ImgLieResnet, num_epochs=100,
                bs=50, lr=3e-3, aug=False,#True, 
                optim=Adam, device='cuda', trainer=Classifier,
                split={'train':40000}, small_test=False, net_config={}, opt_config={},
                trainer_config={'log_dir':None}):

    # Prep the datasets splits, model, and dataloaders
    datasets = split_dataset(dataset(f'~/datasets/{dataset}/'),splits=split)
    datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)
    device = torch.device(device)
    model = network(num_targets=datasets['train'].num_targets,**net_config).to(device)
    if aug: model = torch.nn.Sequential(datasets['train'].default_aug_layers(),model)
    model,bs = try_multigpu_parallelize(model,bs)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    if small_test: dataloaders['test'] = islice(dataloaders['test'],1+len(dataloaders['train'])//10)
    # Add some extra defaults if SGD is chosen
    opt_constr = partial(optim, lr=lr, **opt_config)
    lr_sched = cosLr(num_epochs)
    return trainer(model,dataloaders,opt_constr,lr_sched,**trainer_config)



import argparse
parser = argparse.ArgumentParser(description='LieConv Tests')
parser.add_argument('--rot', type=int, default=1, metavar='N',
                    help='rotated CIFAR100 (default: False)')
parser.add_argument('--scr', type=int, default=0, metavar='N',
                    help='scramble (default: False)')
parser.add_argument('--ker', type=int, default=128, metavar='N',
                    help='k in LieConv layer (default: 128)')
parser.add_argument('--nlay', type=int, default=2, metavar='N',
                    help='number of layers (default: 2)')
parser.add_argument('--epochs', type=int, default=40, metavar='N',
                    help='number of epochs to train (default: 40)')
parser.add_argument('--lr', type=float, default=3e-3, metavar='N',
                    help='learning rate (default: 3e-3)')
parser.add_argument('--bn', type=int, default=1, metavar='N',
                    help='batch normalization (default: True)')



args = parser.parse_args()

SCRAMBLE = args.scr
ROTATE = args.rot

ker = args.ker
nlay = args.nlay
batchnorm = bool(args.bn)
EPOCHS = args.epochs




Trial = train_trial(makeTrainer)
defaults = copy.deepcopy(makeTrainer.__kwdefaults__)

if ROTATE:
    defaults['dataset'] = RotCIFAR100 #MnistRotScrambleDataset
elif SCRAMBLE:
    defaults['dataset'] = RotScramCIFAR100
else:    
    print("=============\n\n Using default CIFAR100\n\n=============")
    defaults['dataset'] = CIFAR100
    
    
defaults['net_config'] = dict(chin=3, 
                              num_layers=nlay, 
                              k=ker,
                              bn= batchnorm
                             )
defaults['num_epochs'] = EPOCHS
defaults['lr'] = args.lr

print(defaults)
fnam = f'./results/lie_conv-cifar100{"-rot" if ROTATE else ""}{"-scr" if SCRAMBLE else ""}-lay{nlay}-k{ker}.pkl'

print('\n', fnam,'\n')

results = Trial(defaults)

net = ImgLieResnet(**defaults['net_config'])
param_size_list = [(name,tuple(param.shape)) for name, param in net.net.named_parameters() if param.requires_grad]

out = dict( net_configs = results[0],
    results = results[1].to_dict(),
    params = param_size_list,
    total_params = sum([np.prod(i[1]) for i in param_size_list])
          )

print('# params: ', out['total_params'])
import pickle 
pickle.dump(out, open(fnam, 'wb'))



# if __name__=="__main__":
#     Trial = train_trial(makeTrainer)
#     defaults = copy.deepcopy(makeTrainer.__kwdefaults__)
#     defaults['save'] = False
#     Trial(argupdated_config(defaults,namespace=(lieConv,lieGroups)))
