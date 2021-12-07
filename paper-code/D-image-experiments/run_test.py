#!/usr/bin/env python

import lconv
import tensorflow as tf

sess = tf.compat.v1.InteractiveSession()
# K = tf.keras.backend

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Layer, Input, Flatten, Reshape, Dense, Conv2D, MaxPool2D

import pickle as pk
import json
import numpy as np


from scipy import ndimage

def rotated_ims_rand(x):
    return np.float32([ndimage.rotate(i, (np.random.rand()-.5)*180, reshape=False, mode='nearest') for i in x])
        
class Scramble_x:
    def __init__(self,x):
        s = x.shape[1:-1]
        self.idx = np.argsort(np.random.rand(np.prod(s)))
        r,c = np.int0(self.idx/s[0]), (self.idx % s[1]) 
        self.x = np.float32([i[r,c].reshape(s+(x.shape[-1],)) for i in x])


# Defaults        
configs= {
    'dataset': dict(name='mnist' ,#'mnist',cifar100, 
                    rotate=False, 
                    scramble=False,),
    'net': dict(architecture= 'lconv', # 'cnn', 'fc', 
                num_filters=32, 
                kernel_size=9, 
                L_hid= [8], #[16], 
                activation = 'relu',
                L_trainable = True),
}


import argparse

parser = argparse.ArgumentParser(description='Run experiments using L-conv or baselines.')
# Add the arguments
# parser.add_argument('dataset_name',#action='store',
# #                    metavar='--data',
#                    type=str, #required=True,
#                    help='mnist, cifar10, cifar100')

# Add the arguments
# parser.add_argument('config_json_file', #action='store',
# #                        metavar='--config',
#                        type=str, #required=True,
#                        help='json file of model configuration')
parser.add_argument('--architecture', action='store', type=str, help="lconv (default), cnn, fc")
parser.add_argument('--dataset', action='store', type=str, help='mnist (default), cifar10, cifar100, fashion_mnist')
parser.add_argument('--rotate', action='store_const', const=True)
parser.add_argument('--scramble', action='store_const', const=True)
parser.add_argument('--lrand', action='store_true', help='Whether L are random or trainable')
parser.add_argument('--epochs', action='store', type=int)
parser.add_argument('--hid', action='store', type=int, help='# hidden units in L for low-rank encoding')

# parser.add_argument('--test', action='store', type=bool, required=False)

args = parser.parse_args()
# print(args.architecture)
# exit()

# print(args.dataset_name)
# print(args.config_json_file)
# configs = json.load(open(args.config_json_file,'r'))

if args.architecture:
    configs['net']['architecture'] = args.architecture

if args.dataset:
    configs['dataset']['name'] = args.dataset
    
if  args.rotate!=None:
    configs['dataset']['rotate'] = args.rotate
    
if  args.scramble!=None:
    configs['dataset']['scramble'] = args.scramble 
    

print(args.lrand)

if args.lrand:
    configs['net']['L_trainable'] = False

if args.hid:
    configs['net']['L_hid'] = [args.hid]

    
EPOCHS = args.epochs or 30

print(configs)

# exit()

# cf: conv filters, ker: kernel size, d: dense (FC) units 
# make_model_name = lambda cf,ker, d: ''.join([('c%s(k%d)' %(cf,ker) if len(cf) else ''),('d%s'%d if len(d) else '')]) or 'base'
# cf: conv filters, ker: kernel size, d: dense (FC) units 



dataset_name = configs['dataset']['name']

dataset = eval("tf.keras.datasets.%s.load_data()" %dataset_name) 
(x_train, y_train), (x_test,y_test) = dataset
if len(x_train.shape) == 3:
    # mnist channel is missing
    x_train = x_train[...,np.newaxis]
    
# normalize
x_train = x_train/x_train[:100].max() -.5 
# make categorical
y_train = tf.keras.utils.to_categorical(y_train)


results = {'configs':configs,}

if configs['dataset']['rotate']:
    print('Rotating images')
    x_train = rotated_ims_rand(x_train)
    
if configs['dataset']['scramble']:
    print('Scrambling images')
    scr = Scramble_x(x_train)
    x_train = scr.x
    results['scramble_idx']=scr.idx.tolist()

##### Make model #####

net = configs['net']
# arch = net['architecture']

kernel_size = net['kernel_size']
# L_hid = net['L_hid']
# activation = net['activation']
# L_trainable = net['L_trainable']


inp = Input(x_train[0].shape)

if net['architecture']=='lconv':
    x = tf.reshape(inp, shape=(-1,np.prod(inp.shape[1:-1]), inp.shape[-1]))
    lay = lconv.L_Conv(num_filters= net['num_filters'], 
                          kernel_size= kernel_size, 
                          L_hid = net['L_hid'], 
                          activation = net['activation'],)

    x = lay(x)
    lay.L.trainable = net['L_trainable']
    
elif net['architecture']=='cnn':
    kx = int(round(np.sqrt(kernel_size)))
    ky = int(round(kernel_size/kx))
    kernel_size = (kx,ky)
    x = Conv2D(filters=net['num_filters'], kernel_size=kernel_size, activation = net['activation'])(inp)
    # x = cnn(inp)
elif net['architecture']=='fc':
    k = net['kernel_size']
    nf = net['num_filters']
    hid = net['L_hid'][0]
    act = net['activation']
    xs = np.prod(inp.shape[1:-1])
    
    x = Flatten()(inp)
    # FC comparable to L-conv, but no shared weights 
    x = Dense(k*hid, activation = act)(x)
    x = Dense(xs*nf, activation = act)(x)

if net['architecture']!='fc':
    x = Flatten()(x)

# x = Dense(100, activation = 'relu')(x)

out = Dense(y_train.shape[-1], activation='softmax')(x)

model = Model(inputs = [inp], outputs = [out])
model.compile(loss = tf.keras.losses.categorical_crossentropy, metrics = ['accuracy'])

model.summary()



##### Train model

h = model.fit(x_train, y_train, validation_split=0.2, epochs=EPOCHS)

##### model name and results
non_trainable = 0
if net['architecture'] == 'lconv':
    model_name = f"L-conv-nf{net['num_filters']}-hid{net['L_hid']}-L_trainable{net['L_trainable']}-ker{kernel_size}" 
    non_trainable = (0 if net['L_trainable'] else lay.L.count_params())
elif net['architecture'] == 'cnn':
    model_name = f"CNN-nf{net['num_filters']}-ker{kernel_size}"
elif net['architecture'] =='fc': 
    model_name = f"FC-nf{net['num_filters']}-hid{net['L_hid']}-ker{kernel_size}" 

    
model_name += f"-act-{net['activation']}"
num_params = model.count_params() - non_trainable

out_file_name = f"./results/{dataset_name}/{model_name}-rotate={configs['dataset']['rotate']}-scrambled={configs['dataset']['scramble']}.json"

results = {}
results.update({
    'num_params':num_params,
    'result':h.history,
#     'result': {k: np.float32(v).tolist() for k,v in h.history.items()}, # bug in json or TF2
          })

# for k,v in results['result'].items():
#     print(k,type(v))

import os

# print(h.history)

dirs = os.path.split(out_file_name)[0]
os.makedirs(dirs,exist_ok=True)

print(out_file_name)


json.dump(results, open(out_file_name, 'w'))