# -*- coding:utf-8 -*-
import torch
import numpy as np
from torch.optim import SGD, ASGD
from torch.autograd import Variable
import torch.nn.functional as F


class SkipGram(torch.nn.Module):
    """A SkipGram model implemented using torch"""
    def __init__(self, num_nodes, dimensions):
        super(SkipGram, self).__init__()
        initial_matrix = np.random.uniform(
            low=-0.5/dimensions,
            high=0.5/dimensions,
            size=(num_nodes, dimensions),
        )
        self.EmMatrix_u = torch.nn.Embedding(num_nodes, dimensions, sparse=False,
                                             _weight=torch.from_numpy(initial_matrix))
        self.EmMatrix_v = torch.nn.Embedding(num_nodes, dimensions, sparse=False,
                                             _weight=torch.from_numpy(initial_matrix))
        self.num_nodes = num_nodes

    def forward(self, u, v, n_negs):
        """
        :param u: like [1, 2, 3]
        :param v: like [2, 3, 4]
        :param n_negs: int, number of negative samples
        :return:
        """
        u = torch.LongTensor(u)
        v = torch.LongTensor(v)
        ns = torch.FloatTensor(n_negs).uniform_(0, self.num_nodes-1).long()
        uvectors = self.EmMatrix_u(u)
        vvectors = self.EmMatrix_v(v).t()
        nvectors = self.EmMatrix_v(ns).neg().t()
        posloss = F.logsigmoid(torch.mm(uvectors, vvectors)).mean(1)
        negloss = F.logsigmoid(torch.mm(uvectors, nvectors)).mean(1)
        return -(posloss+negloss).mean()

    def out_embeddings(self):
        """
        :param path: File name
        :param id2node: dict, a mapping from node index to node name
        :return:
        """
        return self.EmMatrix_u.weight.data.numpy()

    def fit(self, us, vs, n_negs, initial_lr=0.1, max_iters=3):
        lr = initial_lr
        optimizer = SGD(params=self.parameters(), lr=lr)
        for j in range(len(us)):
            u = us[j]
            v = vs[j]
            for i in range(max_iters):
                optimizer.zero_grad()
                loss = self.forward(u, v, n_negs)
                loss.backward()
                optimizer.step()
