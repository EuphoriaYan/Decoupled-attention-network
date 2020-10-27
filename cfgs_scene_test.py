# coding:utf-8
import torch
import torch.optim as optim
import os
from dataset_scene import *
from torchvision import transforms
from DAN import *

global_cfgs = {
    'state': 'Test',
    'epoch': 30,
    'show_interval': 50,
    'test_interval': 1000
}

dataset_cfgs = {
    'dataset_train': lmdbDataset,
    'dataset_train_args': {
        'roots': ['dataset/line_train/book_gen_20201021'],
        'img_height': 128,
        'img_width': 32,
        # 'transform': transforms.Compose([transforms.Resize(128, 32), transforms.ToTensor()]),
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Train',
    },
    'dataloader_train': {
        'batch_size': 64,
        'shuffle': True,
        'num_workers': 4,
    },

    'dataset_test': lmdbDataset,
    'dataset_test_args': {
        'roots': ['dataset/line_val'],
        'img_height': 128,
        'img_width': 32,
        # 'transform': transforms.Compose([transforms.Resize(128, 32), transforms.ToTensor()]),
        'transform': transforms.Compose([transforms.ToTensor()]),
        'global_state': 'Test',
    },
    'dataloader_test': {
        'batch_size': 64,
        'shuffle': False,
        'num_workers': 4,
    },

    'case_sensitive': False,
    # 'dict_dir': 'dict/dic_36.txt'
    'dict_dir': 'dict/charset_xl.txt'
}

net_cfgs = {
    'FE': Feature_Extractor,
    'FE_args': {
        'strides': [(1, 1), (2, 2), (1, 1), (2, 2), (1, 1), (1, 1)],
        'compress_layer': False,
        # 'input_shape': [1, 32, 128],  # C x H x W
        'input_shape': [1, 128, 32],  # C x H x W
    },
    'CAM': CAM,
    'CAM_args': {
        'maxT': 35,
        'depth': 8,
        'num_channels': 64,
    },
    'DTD': DTD,
    'DTD_args': {
        # 'nclass': 38,  # extra 2 classes for Unkonwn and End-token
        'nclass': 23325,  # extra 2 classes for Unkonwn and End-token
        'nchannel': 512,
        'dropout': 0.3,
    },

    'init_state_dict_fe': 'models/scene/exp1/E29_I1800-2397_M0.pth',
    'init_state_dict_cam': 'models/scene/exp1/E29_I1800-2397_M1.pth',
    'init_state_dict_dtd': 'models/scene/exp1/E29_I1800-2397_M2.pth',

    # 'init_state_dict_fe': None,
    # 'init_state_dict_cam': None,
    # 'init_state_dict_dtd': None,

}

optimizer_cfgs = {
    # optim for FE
    'optimizer_0': optim.AdamW,
    'optimizer_0_args': {
        'lr': 3e-4,
    },

    'optimizer_0_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_0_scheduler_args': {
        'milestones': [3, 5],
        'gamma': 0.7,
    },

    # optim for CAM
    'optimizer_1': optim.AdamW,
    'optimizer_1_args': {
        'lr': 3e-4,
    },
    'optimizer_1_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_1_scheduler_args': {
        'milestones': [3, 5],
        'gamma': 0.7,
    },

    # optim for DTD
    'optimizer_2': optim.AdamW,
    'optimizer_2_args': {
        'lr': 3e-4,
    },
    'optimizer_2_scheduler': optim.lr_scheduler.MultiStepLR,
    'optimizer_2_scheduler_args': {
        'milestones': [3, 5],
        'gamma': 0.7,
    },
}

saving_cfgs = {
    'saving_iter_interval': 900,
    'saving_epoch_interval': 1,
    'saving_path': 'models/scene/exp1',
}


def showcfgs(s):
    for key in s.keys():
        print(key, s[key])
    print('')


os.makedirs(saving_cfgs['saving_path'], exist_ok=True)
