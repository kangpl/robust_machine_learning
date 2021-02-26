from __future__ import print_function, division

import numpy as np
import torch
from torch.utils.data import Dataset


class Cifar10Dataset(Dataset):
    """cifar10 with perturbation dataset."""

    def __init__(self, dataPath, perturbationPath, labelPath):
        self.data = np.load(dataPath)
        self.perturbation = np.load(perturbationPath)
        self.targets = np.load(labelPath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, perturbation, target = self.data[index], self.perturbation[index], self.targets[index]
        img = torch.from_numpy(img)
        perturbation = torch.from_numpy(perturbation)
        return img, perturbation, target
