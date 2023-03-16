"""
Toy Example with Bayesian Softmax Regression
"""
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal, MultivariateNormal, Categorical, kl_divergence
from torch.distributions.kl import kl_divergence as kl

from tqdm import tqdm

import seaborn as sns

#%%

class ToyDataset(Dataset):
    def __init__(self, n_samples=1000, n_classes=3):
        self.n_samples = n_samples
        self.n_classes = n_classes

        self.X = torch.randn(n_samples, 512)
        self.Y = torch.randint(0, n_classes, (n_samples,))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

#%%

class ToyModel(nn.Module):
    def __init__(self, n_classes=3):
        super().__init__()
        self.n_classes = n_classes

        self.fc1 = nn.Linear(512, n_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = F.softmax(x, dim=1)
        return x

#%%

# Create dataset
dataset = ToyDataset(n_samples=1000, n_classes=3)

#%%

# Draw 512x3 samples from a multivariate normal distribution
# with mean 0 and covariance matrix I
# and then apply softmax to each row
# to get a probability distribution
# over the 3 classes
W = torch.randn(10000, 512, 3)
SM_X = F.softmax(dataset[1][0]@W, dim=0)

factor = (1-SM_X[:,0])**2 + (0-SM_X[:,1])**2 + (0-SM_X[:,2])**2

#%% Plot factor histogram

plt.figure(figsize=(8, 8/1.618))
plt.hist(factor.numpy(), bins=100)
plt.show()

#%%

# Plot SM_X using barycentric coordinates
# https://en.wikipedia.org/wiki/Barycentric_coordinate_system

# Create a triangle
x = torch.tensor([0, 1, 0, 0])
y = torch.tensor([0, 0, 1, 0])

# Plot the triangle
plt.figure(figsize=(8, 8/1.618))
# plt.plot(x, y, 'k-')
# plt.plot(x, y, 'ko')

# Plot the barycentric coordinates
plt.scatter(SM_X[:,0].numpy(), SM_X[:, 1].numpy())

plt.show()

#%%
