import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import copy

class ManifoldMixupDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        new_idx = np.random.randint(0, len(self.dataset))
        x_0, y_0 = self.dataset[index]
        x_1, y_1 = self.dataset[new_idx]
        return [x_0, x_1], [y_0, y_1]

    def __len__(self):
        return len(self.dataset)

class ManifoldMixupModel(nn.Module):

    def __init__(self, model, alpha=1.0, interpolation_adv=False):
        super(ManifoldMixupModel, self).__init__()
        self.model = model
        self.alpha = alpha
        self.intermediate_other = None
        self.lam = None
        self.interpolation_adv = interpolation_adv

    def forward(self, x, switch_adv=False):
        x_0, x_1 = x
        self.lam = np.random.beta(self.alpha, self.alpha)
        k = np.random.randint(-1, len(list(self.model.modules())))
        if k == -1:
            x_ = self.lam * x_0 + (1 - self.lam) * x_1
            out = self.model(x_)
        else:
            fetcher_hook = list(self.model.modules())[k].register_forward_hook(self.hook_fetch)
            self.model(x_1)
            fetcher_hook.remove()
            modifier_hook = list(self.model.modules())[k].register_forward_hook(self.hook_modify)
            out = self.model(x_0)
            modifier_hook.remove()
        if self.interpolation_adv and not switch_adv:
            return out, x, self, self.lam
        return out, self.lam

    def hook_modify(self, module, input, output):
        output = (1 - self.lam) * self.intermediate_other + self.lam * output

    def hook_fetch(self, module, input, output):
        self.intermediate_other = output

class ManifoldMixupLoss(nn.Module):

    def __init__(self, criterion):
        super(ManifoldMixupLoss, self).__init__()
        self.criterion = criterion

    def forward(self, outs, y):
        out, lam = outs
        y_0, y_1 = y
        loss_0, loss_1 = self.criterion(out, y_0), self.criterion(out, y_1)
        return lam * loss_0 + (1 - lam) * loss_1
        