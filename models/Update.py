# Modified from: https://github.com/lgcollins/FedRep by Liam Collins et al.

# For MAML (PerFedAvg) implementation, code was adapted from https://github.com/AntreasAntoniou/HowToTrainYourMAMLPytorch/blob/master/few_shot_learning_system.py
# credit: Antreas Antoniou

# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import enum
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import math
import numpy as np
import time
import copy
import FedProx

from models.test import test_img_local
from models.language_utils import get_word_emb_arr, repackage_hidden, process_x, process_y
from models.bsml import balanced_softmax_loss

frozenKey = ['fc4.weight', 'fc4.bias','fe_fc1.weight', 'fe_fc1.bias','fe_fc2.weight', 'fe_fc2.bias','fe_conv2.weight', 'fe_conv2.bias','fe_conv1.weight', 'fe_conv1.bias']
feKey = ['fc4.weight', 'fc4.bias','fe_fc1.weight', 'fe_fc1.bias','fe_fc2.weight', 'fe_fc2.bias','fe_conv2.weight', 'fe_conv2.bias','fe_conv1.weight', 'fe_conv1.bias']


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs, name=None):
        self.dataset = dataset
        self.idxs = list(idxs)
        self.name = name

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        if self.name is None:
            image, label = self.dataset[self.idxs[item]]
        elif 'femnist' in self.name:
            image = torch.reshape(torch.tensor(self.dataset['x'][item]), (1, 28, 28))
            label = torch.tensor(self.dataset['y'][item])
        elif 'sent140' in self.name:
            image = self.dataset['x'][item]
            label = self.dataset['y'][item]
        else:
            image, label = self.dataset[self.idxs[item]]
        return image, label

# Generic local update class, implements local updates for FedRep, FedPer, LG-FedAvg, FedAvg, FedProx
class LocalUpdate(object):
    def __init__(self, args, dataset=None, idxs=None, indd=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        if 'femnist' in args.dataset or 'sent140' in args.dataset:
            self.ldr_train = DataLoader(DatasetSplit(dataset, np.ones(len(dataset['x'])), name=self.args.dataset),
                                        batch_size=self.args.local_bs, shuffle=True)
        else:
            self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

        if 'sent140' in self.args.dataset and indd == None:
            VOCAB_DIR = 'models/embs.json'
            _, self.indd, vocab = get_word_emb_arr(VOCAB_DIR)
            self.vocab_size = len(vocab)
        elif indd is not None:
            self.indd = indd
        else:
            self.indd = None

        self.dataset = dataset
        self.idxs = idxs

    def train(self, net, w_glob_keys, last=False, dataset_test=None, ind=-1, idx=-1, lr=0.1):
        bias_p = []
        weight_p = []
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        optimizer = torch.optim.SGD(
            [
                {'params': weight_p, 'weight_decay': 0.0001},
                {'params': bias_p, 'weight_decay': 0}
            ],
            lr=lr, momentum=0.5
        )
        if self.args.alg == 'prox':
            optimizer = FedProx.FedProx(net.parameters(),
                                        lr=lr,
                                        gmf=self.args.gmf,
                                        mu=self.args.mu,
                                        ratio=1 / self.args.num_users,
                                        momentum=0.5,
                                        nesterov=False,
                                        weight_decay=1e-4)

        local_eps = self.args.local_ep
        if last:
            if self.args.alg == 'fedavg' or self.args.alg == 'prox':
                local_eps = 10
                net_keys = [*net.state_dict().keys()]
                if 'cifar' in self.args.dataset:
                    w_glob_keys = [net.weight_keys[i] for i in [0, 1, 3, 4]]
                elif 'sent140' in self.args.dataset:
                    w_glob_keys = [net_keys[i] for i in [0, 1, 2, 3, 4, 5]]
                elif 'mnist' in self.args.dataset:
                    w_glob_keys = [net_glob.weight_keys[i] for i in [0, 1, 2]]
            elif 'maml' in self.args.alg:
                local_eps = 5
                w_glob_keys = []
            else:
                local_eps = max(10, local_eps - self.args.local_rep_ep)

        if self.args.dataset == 'cifar10':
            num_classes = 10
        elif self.args.dataset == 'cifar100':
            num_classes = 100
        class_count = torch.zeros(num_classes, 1)
        with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
            rand_set_all = torch.load(f)
        for j in range(num_classes):
            if j in rand_set_all[self.indd]:
                class_count[j][0] = 1
        head_eps = local_eps - self.args.local_rep_ep
        epoch_loss = []
        num_updates = 0
        if 'sent140' in self.args.dataset:
            hidden_train = net.init_hidden(self.args.local_bs)
        for iter in range(local_eps):
            done = False

            # for FedRep, first do local epochs for the head
            if (iter < head_eps and self.args.alg == 'fedrep') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True

            # then do local epochs for the representation
            elif iter >= head_eps and self.args.alg == 'fedrep' and not last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = True
                    else:
                        param.requires_grad = False

            elif (iter < head_eps and self.args.alg == 'flute') or last:
                for name, param in net.named_parameters():
                    if name in w_glob_keys:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True


            elif iter >= head_eps and self.args.alg == 'flute' and not last:
                for name, param in net.named_parameters():
                    param.requires_grad = True
                    # if name in w_glob_keys:
                    #     param.requires_grad = True
                    # else:
                    #     param.requires_grad = False
            


            elif self.args.alg != 'fedrep' and self.args.alg != 'flute':
                for name, param in net.named_parameters():
                    param.requires_grad = True

            # if self.args.dataset == 'cifar10':
            #     k = 64
            # elif self.args.dataset == 'cifar100':
            #     k = 128

            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                if 'sent140' in self.args.dataset:
                    input_data, target_data = process_x(images, self.indd), process_y(labels, self.indd)
                    if self.args.local_bs != 1 and input_data.shape[0] != self.args.local_bs:
                        break
                    net.train()
                    data, targets = torch.from_numpy(input_data).to(self.args.device), torch.from_numpy(target_data).to(
                        self.args.device)
                    net.zero_grad()
                    hidden_train = repackage_hidden(hidden_train)
                    output, hidden_train = net(data, hidden_train)
                    loss = self.loss_func(output.t(), torch.max(targets, 1)[1])

                    loss.backward()
                    optimizer.step()
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)

                net.zero_grad()
                log_probs, feature = net(images)
                loss = self.loss_func(log_probs, labels)
                if self.args.alg == 'flute' and iter >= head_eps:
                    for name, param in net.named_parameters():
                        if name == 'fc3.weight':
                            f_norm = torch.norm(torch.matmul(param, param.t()), p='fro')
                            r1 = 0.25 * torch.norm(torch.matmul(param, param.t()) / f_norm - 1 / torch.sqrt(
                                torch.tensor(num_classes - 1).to(self.args.device)) \
                                                    * torch.mul((torch.eye(num_classes).to(
                                self.args.device) - 1 / num_classes * torch.ones((num_classes, num_classes)).to(
                                self.args.device)), torch.matmul(class_count, class_count.T).to(self.args.device)), p='fro')
                            loss += r1
                    for name, param in net.named_parameters():
                        if name == 'fc3.weight':
                            r2 = 0.0025 * torch.norm(param, p='fro') ** 2
                            loss += r2
                    r3 = 0.0005 * torch.norm(feature, p='fro') ** 2
                    loss += r3
                elif self.args.alg == 'flute' and iter < head_eps:
                    for name, param in net.named_parameters():
                        if name == 'fc3.weight':
                            f_norm = torch.norm(torch.matmul(param, param.t()), p='fro')
                            r1 = 0.25 * torch.norm(torch.matmul(param, param.t()) / f_norm - 1 / torch.sqrt(
                                torch.tensor(num_classes - 1).to(self.args.device)) \
                                                    * torch.mul((torch.eye(num_classes).to(
                                self.args.device) - 1 / num_classes * torch.ones((num_classes, num_classes)).to(
                                self.args.device)), torch.matmul(class_count, class_count.T).to(self.args.device)), p='fro')
                            loss += r1
                    for name, param in net.named_parameters():
                        if name == 'fc3.weight':
                            r2 = 0.0025 * torch.norm(param, p='fro') ** 2
                            loss += r2
                loss.backward()
                optimizer.step()


                num_updates += 1
                batch_loss.append(loss.item())

                if num_updates == self.args.local_updates:
                    done = True
                    break
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            if done:
                break

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), self.indd


class ServerUpdateNC2(object):

    def __init__(self, args, w_locals=None, idxs_users=None):
        self.args = args
        self.w_locals = w_locals
        self.idxs_users = idxs_users

    def train(self, gamma1=1, gamma2=1, lr=0.5):
        w_locals = self.w_locals

        idxs_users = self.idxs_users

        num_classes = self.args.num_classes

        num_users = len(idxs_users)

        class_count = torch.zeros(num_classes)
        with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
            rand_set_all = torch.load(f)
        for i in idxs_users:
            for j in range(num_classes):
                if j in rand_set_all[i]:
                    class_count[j] += 1

        not_pass = True
        # if 0 in class_count:
        #     not_pass = False
        #     print('**0 in class count, pass server side optimization.**')

        if not_pass:
            if self.args.dataset == 'cifar10':
                k = 64
            elif self.args.dataset == 'cifar100':
                k = 128

            w_mat = torch.rand((k, num_classes * num_users))

            b_mat = torch.rand((num_classes, num_users))

            for idx_iter, idx in enumerate(idxs_users):
                for col in range(k):
                    for row in range(num_classes):
                        w_mat[col][row + idx_iter * num_classes] = w_locals[idx]['fc3.weight'][row][col]

                for row in range(num_classes):
                    b_mat[row][idx_iter] = w_locals[idx]['fc3.bias'][row]

            w_mat.requires_grad_(True)
            w_mat.retain_grad()
            b_mat.requires_grad_(True)
            b_mat.retain_grad()

            # optimizer = torch.optim.SGD([w_mat, b_mat], lr = 0.5)
            optimizer = torch.optim.SGD([w_mat], lr=0.5)
            optimizer.zero_grad()
            loss = 0

            with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
                rand_set_all = torch.load(f)
            for idx, i in enumerate(idxs_users):
                class_count = torch.zeros(num_classes, 1)
                for j in range(num_classes):
                    if j in rand_set_all[i]:
                        class_count[j][0] = 1

                local_mat = torch.zeros(k, num_classes)
                for j in range(num_classes):
                    local_mat[:, j] = w_mat[:, idx * num_classes + j]

                f_norm = torch.norm(torch.matmul(local_mat.T, local_mat), p='fro')
                loss += 1 / num_users * torch.norm(
                    torch.matmul(local_mat.T, local_mat) / f_norm - 1 / torch.sqrt(torch.tensor(num_classes - 1)) \
                    * torch.mul((torch.eye(num_classes) - 1 / num_classes * torch.ones((num_classes, num_classes))), \
                                torch.matmul(class_count, class_count.T)), p='fro')

            loss.backward()

            optimizer.step()

            for idx_iter, idx in enumerate(idxs_users):
                for col in range(k):
                    for row in range(num_classes):
                        w_locals[idx]['fc3.weight'][row][col] = w_mat[col][row + idx_iter * num_classes]

                for row in range(num_classes):
                    w_locals[idx]['fc3.bias'][row] = b_mat[row][idx_iter]

        return w_locals


def fun_reg(args, w_locals):
    if args.dataset == 'cifar10':
        k = 64
    elif args.dataset == 'cifar100':
        k = 128
    num_classes = args.num_classes
    num_users = args.num_users
    idxs_users = range(num_users)
    shard = args.shard_per_user

    class_count = torch.zeros(num_classes)
    with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
        rand_set_all = torch.load(f)
    for i in idxs_users:
        for j in range(num_classes):
            if j in rand_set_all[i]:
                class_count[j] += 1

    w_mat = torch.rand((k, num_classes * num_users))

    b_mat = torch.rand((num_classes, num_users))

    for idx_iter, idx in enumerate(idxs_users):
        for col in range(k):
            for row in range(num_classes):
                w_mat[col][row + idx_iter * num_classes] = w_locals[idx]['fc3.weight'][row][col]

        for row in range(num_classes):
            b_mat[row][idx_iter] = w_locals[idx]['fc3.bias'][row]

    loss = 0
    lloss = 0

    with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
        rand_set_all = torch.load(f)
    for idx, i in enumerate(idxs_users):
        class_count = torch.zeros(num_classes, 1)
        for j in range(num_classes):
            if j in rand_set_all[i]:
                class_count[j][0] = 1

        local_mat = torch.zeros(k, num_classes)
        for j in range(num_classes):
            local_mat[:, j] = w_mat[:, idx * num_classes + j]

        f_norm = torch.norm(torch.matmul(local_mat.T, local_mat), p='fro')

        loss += 1 / num_users * torch.norm(
            torch.matmul(local_mat.T, local_mat) / f_norm - 1 / torch.sqrt(torch.tensor(num_classes - 1)) \
            * torch.mul((torch.eye(num_classes) - 1 / num_classes * torch.ones((num_classes, num_classes))), \
                        torch.matmul(class_count, class_count.T)), p='fro')

        lloss += 1 / num_users * torch.norm(
            torch.matmul(local_mat.T, local_mat) / f_norm - 1 / torch.sqrt(torch.tensor(shard - 1)) \
            * torch.mul((torch.eye(num_classes) - 1 / shard * torch.ones((num_classes, num_classes))), \
                        torch.matmul(class_count, class_count.T)), p='fro')


    print('****Global Regularization: ' + str(loss.item()) + ' ,****Local Regularization: ' + str(lloss.item()) + '\n')

    return loss.item(), lloss.item()


def fun_reg_2(args, w_locals, idxs_users):
    num_classes = args.num_classes

    num_users = len(idxs_users)

    class_count = torch.zeros(num_classes)
    with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
        rand_set_all = torch.load(f)
    for i in idxs_users:
        for j in range(num_classes):
            if j in rand_set_all[i]:
                class_count[j] += 1

    not_pass = True

    if not_pass:
        if args.dataset == 'cifar10':
            k = 64
        elif args.dataset == 'cifar100':
            k = 128

        w_mat = torch.rand((k, num_classes * num_users))

        b_mat = torch.rand((num_classes, num_users))

        for idx_iter, idx in enumerate(idxs_users):
            for col in range(k):
                for row in range(num_classes):
                    w_mat[col][row + idx_iter * num_classes] = w_locals[idx]['fc3.weight'][row][col]

            for row in range(num_classes):
                b_mat[row][idx_iter] = w_locals[idx]['fc3.bias'][row]

        loss = 0

        with open("/home/anaconda/content/flute/save/rand_set_all.txt", 'rb') as f:
            rand_set_all = torch.load(f)
        for idx, i in enumerate(idxs_users):
            class_count = torch.zeros(num_classes, 1)
            for j in range(num_classes):
                if j in rand_set_all[i]:
                    class_count[j][0] = 1

            local_mat = torch.zeros(k, num_classes)
            for j in range(num_classes):
                local_mat[:, j] = w_mat[:, idx * num_classes + j]

            f_norm = torch.norm(torch.matmul(local_mat.T, local_mat), p='fro')
            loss += 1 / num_users * torch.norm(
                torch.matmul(local_mat.T, local_mat) / f_norm - 1 / torch.sqrt(torch.tensor(num_classes - 1)) \
                * torch.mul((torch.eye(num_classes) - 1 / num_classes * torch.ones((num_classes, num_classes))), \
                            torch.matmul(class_count, class_count.T)), p='fro')

    print('****Global Regularization: ' + str(loss.item()) + '\n')

    return loss.item()
