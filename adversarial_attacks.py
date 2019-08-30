from abc import ABC, abstractclassmethod, abstractmethod
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import copy

class BlackBoxAdversarialAttack(ABC):

    def __call__(self, x, y, model):
        model.eval()
        for n, p in model.named_parameters():
            p.requires_grad = False
        x_adv = self.run(x, y, model)
        model.train()
        for n, p in model.named_parameters():
            p.requires_grad = True
        return x_adv

    @abstractmethod
    def run(self, x, y, model):
        pass

class FGSM(BlackBoxAdversarialAttack):

    def __init__(self, adv_criterion, eps=0.25, low=0, high=1):
        self.criterion = adv_criterion
        self.eps = eps
        self.low = low
        self.high = high

    def run(self, x, y, model):
        x_adv = x.clone().detach()
        x_adv.requires_grad = True
        out = model(x_adv)
        loss = self.criterion(out, y)
        loss.backward()
        grad = x_adv.grad.data.sign()
        x_adv = torch.clamp(x_adv + grad * self.eps, self.low, self.high)
        return x_adv.data
        