from abc import ABC, abstractclassmethod, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import copy
from manifold_mixup import ManifoldMixupLoss

class InterpolatedAdversarialLoss(nn.Module):

    def __init__(self, criterion, attack):
        super(InterpolatedAdversarialLoss, self).__init__()
        self.criterion = ManifoldMixupLoss(criterion)
        self.attack = attack

    def forward(self, outs, y):
        out, (x_0, x_1), model, lam = outs
        y_0, y_1 = y
        true_mixup_loss = self.criterion((out, lam), y)
        x_adv_0 = self.attack(x_0, y_0, model.model) 
        x_adv_1 = self.attack(x_1, y_1, model.model)
        outs_adv = model([x_adv_0, x_adv_1], switch_adv=True)
        adv_mixup_loss = self.criterion(outs_adv, y)
        return (true_mixup_loss + adv_mixup_loss) / 2
        