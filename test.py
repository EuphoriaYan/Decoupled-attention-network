# coding:utf-8
# from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
# ------------------------
from utils import *
import cfgs_scene_test as cfgs


# ------------------------
def display_cfgs(models):
    print('global_cfgs')
    cfgs.showcfgs(cfgs.global_cfgs)
    print('dataset_cfgs')
    # cfgs.showcfgs(cfgs.dataset_cfgs)
    print('net_cfgs')
    cfgs.showcfgs(cfgs.net_cfgs)
    print('optimizer_cfgs')
    cfgs.showcfgs(cfgs.optimizer_cfgs)
    print('saving_cfgs')
    cfgs.showcfgs(cfgs.saving_cfgs)
    for model in models:
        print(model)


def flatten_label(target):
    label_flatten = []
    label_length = []
    for i in range(0, target.size()[0]):
        cur_label = target[i].tolist()
        label_flatten += cur_label[:cur_label.index(0) + 1]
        label_length.append(cur_label.index(0) + 1)
    label_flatten = torch.LongTensor(label_flatten)
    label_length = torch.IntTensor(label_length)
    return (label_flatten, label_length)


def Train_or_Eval(models, state='Train'):
    for model in models:
        if state == 'Train':
            model.train()
        else:
            model.eval()


# ---------------------dataset
def load_dataset():
    test_data_set = cfgs.dataset_cfgs['dataset_test'](**cfgs.dataset_cfgs['dataset_test_args'])
    test_loader = DataLoader(test_data_set, **cfgs.dataset_cfgs['dataloader_test'])
    # pdb.set_trace()
    return test_loader


# ---------------------network
def load_network():
    model_fe = cfgs.net_cfgs['FE'](**cfgs.net_cfgs['FE_args'])

    cfgs.net_cfgs['CAM_args']['scales'] = model_fe.Iwantshapes()
    model_cam = cfgs.net_cfgs['CAM'](**cfgs.net_cfgs['CAM_args'])

    model_dtd = cfgs.net_cfgs['DTD'](**cfgs.net_cfgs['DTD_args'])

    if cfgs.net_cfgs['init_state_dict_fe'] is not None:
        model_fe.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_fe']))
    if cfgs.net_cfgs['init_state_dict_cam'] is not None:
        model_cam.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_cam']))
    if cfgs.net_cfgs['init_state_dict_dtd'] is not None:
        model_dtd.load_state_dict(torch.load(cfgs.net_cfgs['init_state_dict_dtd']))

    model_fe.cuda()
    model_cam.cuda()
    model_dtd.cuda()
    return (model_fe, model_cam, model_dtd)


# ---------------------testing stage
def test(test_loader, model, tools):
    Train_or_Eval(model, 'Eval')
    for sample_batched in test_loader:
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)

        data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        target, label_flatten = target.cuda(), label_flatten.cuda()

        features = model[0](data)
        A = model[1](features)
        output, out_length = model[2](features[-1], A, target, length, True)
        prdt_texts, prdt_prob = tools[2].add_iter(output, out_length, length, label)
    tools[2].show()
    Train_or_Eval(model, 'Train')


# ---------------------infer stage
def infer(test_loader, model, tools):
    Train_or_Eval(model, 'Eval')
    total_prdt_texts = []
    total_prdt_prob = []
    for sample_batched in test_loader:
        data = sample_batched['image']
        label = sample_batched['label']
        target = tools[0].encode(label)

        data = data.cuda()
        target = target
        label_flatten, length = tools[1](target)
        target, label_flatten = target.cuda(), label_flatten.cuda()

        features = model[0](data)
        A = model[1](features)
        output, out_length = model[2](features[-1], A, target, length, True)
        prdt_texts, prdt_prob = tools[2].add_iter(output, out_length, length, label)
        total_prdt_texts.extend(prdt_texts)
        total_prdt_prob.extend(prdt_prob.item())
    tools[2].show()
    Train_or_Eval(model, 'Train')
    return total_prdt_texts, total_prdt_prob


# ---------------------------------------------------------
# --------------------------Begin--------------------------
# ---------------------------------------------------------
if __name__ == '__main__':
    # prepare nets, optimizers and data
    model = load_network()
    # display_cfgs(model)
    test_loader = load_dataset()
    print('preparing done')
    # --------------------------------
    # prepare tools
    test_acc_counter = Attention_AR_counter('\ntest accuracy: ', cfgs.dataset_cfgs['dict_dir'],
                                            cfgs.dataset_cfgs['case_sensitive'])
    encdec = cha_encdec(cfgs.dataset_cfgs['dict_dir'], cfgs.dataset_cfgs['case_sensitive'])
    # ---------------------------------
    if cfgs.global_cfgs['state'] == 'Test':
        with torch.no_grad():
            total_prdt_texts, total_prdt_prob = infer(
                (test_loader),
                model,
                [encdec, flatten_label, test_acc_counter]
            )
            for prdt_text, prdt_prob in zip(total_prdt_texts, total_prdt_prob):
                print(prdt_text, '\t', prdt_prob)
    else:
        raise ValueError
    # --------------------------------
