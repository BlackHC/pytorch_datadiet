# -*- coding: utf-8 -*-
"""Cifar10 Data Diet - At Initialization Sampling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1T71vJh7vVVXF1eixIvGr590Wwr7imAE7
"""

#%%

%cd sampled_grad_norms

# Based on https://github.com/y0ast/pytorch-snippets/blob/main/minimal_cifar/train_cifar.py
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from torchvision import models, datasets, transforms

import numpy as np

#%%

def get_CIFAR10(root="./"):
    input_size = 32
    num_classes = 10
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    
    train_transform = transforms.Compose(
        [
            # transforms.RandomCrop(32, padding=4),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    train_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=True, transform=train_transform, download=True
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_dataset = datasets.CIFAR10(
        root + "data/CIFAR10", train=False, transform=test_transform, download=True
    )

    return input_size, num_classes, train_dataset, test_dataset

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(num_classes=10)

        self.resnet.conv1 = torch.nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.resnet.maxpool = torch.nn.Identity()

    def forward(self, x):
        x = self.resnet(x)
        x = F.log_softmax(x, dim=1)

        return x


def train(model, train_loader, optimizer, epoch):
    model.train()

    total_loss = []

    for data, target in tqdm(train_loader):
        data = data.cuda()
        target = target.cuda()

        optimizer.zero_grad()

        prediction = model(data)
        loss = F.nll_loss(prediction, target)

        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

    avg_loss = sum(total_loss) / len(total_loss)
    print(f"Epoch: {epoch}:")
    print(f"Train Set: Average Loss: {avg_loss:.2f}")


def test(model, test_loader):
    model.eval()

    loss = 0
    correct = 0

    for data, target in test_loader:
        with torch.no_grad():
            data = data.cuda()
            target = target.cuda()

            prediction = model(data)
            loss += F.nll_loss(prediction, target, reduction="sum")

            prediction = prediction.max(1)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

    loss /= len(test_loader.dataset)

    percentage_correct = 100.0 * correct / len(test_loader.dataset)

    print(
        "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
            loss, correct, len(test_loader.dataset), percentage_correct
        )
    )

    return loss, percentage_correct

# parse arguments
epochs = 50
lr = 0.05

# set seed
# 1 15 30 45 60 1337
torch.manual_seed(2048)

# get CIFAR10 dataset
input_size, num_classes, train_dataset, test_dataset = get_CIFAR10()

# set dataloaders
kwargs = {"num_workers": 2, "pin_memory": True}

fixed_train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=256, shuffle=False, **kwargs
)
#train_loader = torch.utils.data.DataLoader(
#    train_dataset, batch_size=128, shuffle=True, **kwargs
#)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=256, shuffle=False, **kwargs
)

# create model
#model = Model()

import copy
import gc

#del model_batch
#del model
gc.collect()
torch.cuda.empty_cache()

# model_batch = [copy.deepcopy(model) for _ in range(128*3-1)]
# model_batch += [model]

from functorch.experimental import replace_all_batch_norm_modules_
#replace_all_batch_norm_modules_(model)

from functorch import make_functional_with_buffers, vmap, grad

fmodel, params, buffers = make_functional_with_buffers(model)

def compute_loss_stateless_model (params, buffers, sample, target):
    batch = sample.unsqueeze(0)
    targets = target.unsqueeze(0)

    predictions = fmodel(params, buffers, batch) 
    loss = F.nll_loss(predictions, targets)
    return loss

ft_compute_grad = grad(compute_loss_stateless_model)

ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, None, 0, 0))

def get_grad_norms(params, buffers, fixed_loader):
    fmodel.eval()

    grad_norms = []

    for data_batch, target_batch in tqdm(fixed_loader):
        data_batch = data_batch.cuda()
        target_batch = target_batch.cuda()

        ft_per_sample_grads = ft_compute_sample_grad(params, buffers, data_batch, target_batch)

        squared_norm = 0
        for param_grad in ft_per_sample_grads:
          squared_norm += param_grad.flatten(1).square().sum(dim=-1)
        grad_norms.append(squared_norm.detach().cpu().numpy()**0.5)

    grad_norms = np.concatenate(grad_norms, axis=0)
    return grad_norms

batched_train_grad_norms = []
for i in range(40):
    _, params, buffers = make_functional_with_buffers(Model().cuda())
    grad_norms = get_grad_norms(params, buffers, fixed_train_loader)
    batched_train_grad_norms.append(grad_norms)

batched_test_grad_norms = []
for i in range(1000):
    _, params, buffers = make_functional_with_buffers(Model().cuda())
    grad_norms = get_grad_norms(params, buffers, test_loader)
    batched_test_grad_norms.append(grad_norms)

batched_test_grad_norms_np = np.stack(batched_test_grad_norms, axis=0)

# Save train_grad_norms in file
np.save('batched_test_grad_norms.npy', batched_test_grad_norms_np)

batched_train_grad_norms.extend(list(batched_train_grad_norms_np))

batched_train_grad_norms_np = np.stack(batched_train_grad_norms, axis=0)

batched_train_grad_norms_np.shape

# Save train_grad_norms in file
np.save('batched_train_grad_norms.npy', batched_train_grad_norms_np)

def get_input_norms(fixed_loader):
  input_norms = []

  for data_batch, target_batch in tqdm(fixed_loader):
    # flatten data_batch after the first dimension
    data_batch = data_batch.view(data_batch.shape[0], -1)
    # compute the square norm of the flattened data_batch
    norm = data_batch.square().sum(dim=1)**0.5

    input_norms.append(norm.cpu().numpy())

  input_norms = np.concatenate(input_norms)

  return input_norms

train_input_norms = get_input_norms(fixed_train_loader)

test_input_norms = get_input_norms(test_loader)

# Save train_grad_norms in file
np.save('train_input_norms.npy', train_input_norms)

# Save train_grad_norms in file
np.save('test_input_norms.npy', test_input_norms)

def get_input_stddevs(fixed_loader):
  input_stddevs = []

  for data_batch, target_batch in tqdm(fixed_loader):
    # flatten data_batch after the first dimension
    data_batch = data_batch.view(data_batch.shape[0], -1)
    # compute the square norm of the flattened data_batch
    stddev = data_batch.std(dim=1)

    input_stddevs.append(stddev.cpu().numpy())

  input_stddevs = np.concatenate(input_stddevs)

  return input_stddevs

train_input_stddevs = get_input_stddevs(fixed_train_loader)

test_input_stddevs = get_input_stddevs(test_loader)

# Save train_grad_norms in file
np.save('train_input_stddevs.npy', train_input_stddevs)

# Save train_grad_norms in file
np.save('test_input_stddevs.npy', test_input_stddevs)

#%%

# Load
class TrainStats:
  prefix = 'Train'
  input_norms = np.load('train_input_norms.npy')
  input_stddevs = np.load('train_input_stddevs.npy')
  batched_grad_norms = np.load('batched_train_grad_norms.npy')

  @classmethod
  def compute_grad_norms(cls):
    cls.grad_norms = cls.batched_grad_norms.mean(axis=0)

TrainStats.compute_grad_norms()

# Load
class TestStats(TrainStats):
  prefix = 'Test'
  input_norms = np.load('test_input_norms.npy')
  input_stddevs = np.load('test_input_stddevs.npy')
  batched_grad_norms = np.load('batched_test_grad_norms.npy')

TestStats.compute_grad_norms()

#%%
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")

#%%

import types, inspect

def assign(c):
  def wrapper(f):
    name = f.__name__ if hasattr(f, '__name__') else f.__func__.__name__
    setattr(c, name, f)

  return wrapper

@assign(TrainStats)
@classmethod
def sort_by_input_norms(cls: TrainStats):
  # Sort both arrays by the norm of the gradients
  sorted_indices = np.argsort(cls.input_norms[:len(cls.input_norms)])
  cls.sorted_grad_norms =  cls.grad_norms[sorted_indices]
  cls.sorted_input_stddevs = cls.input_stddevs[sorted_indices]
  cls.sorted_input_norms = cls.input_norms[sorted_indices]

TrainStats.sort_by_input_norms()
TestStats.sort_by_input_norms()

TrainStats.sorted_input_norms.mean(), TrainStats.sorted_grad_norms.mean()

#%%

import matplotlib.pyplot as plt

import scipy.stats as stats

@assign(TrainStats)
@classmethod
def plot(self: TrainStats):
  # Plot
  plt.figure(figsize=(10, 5))
  #plt.plot(self.sorted_input_norms, self.sorted_grad_norms, '.')
  plt.plot(self.sorted_grad_norms, self.sorted_input_norms, '.')
  plt.xlabel('GraNd at init')
  plt.ylabel('Input norm')
  plt.title(f'{self.prefix}: Input norm vs GraNd at Init (Scatter)')
  plt.savefig(f'{self.prefix}_input_norm_vs_grad_norm_scatter.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
  plt.show()

  # # Plot
  # plt.figure(figsize=(10, 5))
  # #plt.plot(self.sorted_input_stddevs, self.sorted_grad_norms, '.')
  # plt.ylabel('Gradient norm')
  # plt.xlabel('Input stddev')
  # plt.title(f'{self.__name__}: Input stddev vs gradient norm')
  # plt.show()

  # Plot both as separate sorted plots
  plt.figure(figsize=(10, 5))
  plt.plot(list(range(len(self.sorted_input_norms))), self.sorted_input_norms, '.', label="Input norm", alpha=0.2,zorder=10)
  plt.plot(list(range(len(self.sorted_grad_norms))), self.sorted_grad_norms, '.', label="GraNd at init", alpha=0.2)
  #plt.ylabel('Norm')
  plt.xlabel('Sorted')
  plt.ylabel('Score')
  plt.title(f'{self.prefix}: Input norm vs GraNd at Init (Sorted)')
  plt.savefig(f'{self.prefix}_input_norm_vs_grad_norm_sorted.svg', format='svg', bbox_inches='tight', pad_inches=0,
              transparent=True)
  plt.legend()
  plt.show()

  ratio = (self.sorted_grad_norms / self.sorted_input_norms)
  # Plot ratio histogram
  plt.figure(figsize=(10, 5))
  _, bins, _ = plt.hist(ratio, bins=100)
  ratio_mean = np.mean(ratio)
  ratio_std = np.std(ratio)  
  print(f'{self.prefix}: Ratio mean: {ratio_mean}, std: {ratio_std}')

  plt.axvline(ratio_mean, color='r', linestyle='--')
  plt.axvline(ratio_mean + ratio_std, color='r', linestyle='--')
  plt.axvline(ratio_mean - ratio_std, color='r', linestyle='--')

  # Compute the probability mass * len(ratio) of the normal variable with same mean and stddev
  # as the ratio - approx just take the pdf of the middle of each bin times bin width
  bin_middle = (bins[1:] + bins[:-1]) / 2
  bin_width = bins[1:] - bins[:-1]  
  expected_counts = stats.norm.pdf(bin_middle, ratio_mean, ratio_std) * bin_width * len(ratio)
  plt.plot(bin_middle, expected_counts)
  plt.xlabel('Exp(Gradient norm / Input norm)')
  plt.ylabel('Count')
  plt.title(f'{self.prefix}: Gradient norm / Input norm')
  plt.savefig(f'{self.prefix}_input_norm_div_grad_norm.svg', format='svg', bbox_inches='tight', pad_inches=0, transparent=True)
  plt.show()

  plt.figure(figsize=(10, 5))
  #plt.plot(list(range(len(self.sorted_input_norms))), self.sorted_input_norms, '.', label="Input norm", alpha=0.2,zorder=10)
  plt.plot(list(range(len(self.sorted_grad_norms))), ratio, '.', label="Ratio", alpha=0.2)
  # Sample from ratio mean and std to generate fake scores
  fake_ratio = np.random.normal(ratio_mean, ratio_std, len(ratio))
  plt.plot(list(range(len(fake_ratio))), fake_ratio, '.', label="Fake Ratio", alpha=0.2)
  #plt.ylabel('Norm')
  plt.xlabel('Sorted')
  plt.ylabel('Score')
  plt.title('Input norm vs gradient norm')
  plt.legend()
  plt.show()

  # Plot sorted ratio and fake_ratio
  plt.figure(figsize=(10, 5))
  plt.plot(list(range(len(self.sorted_grad_norms))), np.sort(ratio), '.', label="Ratio", alpha=0.2)
  plt.plot(list(range(len(self.sorted_grad_norms))), np.sort(fake_ratio), '.', label="Fake Ratio", alpha=0.2)
  plt.xlabel('Individually Sorted')
  plt.ylabel('Ratio')
  plt.title('Ratio vs Fake Ratio')
  plt.legend()
  plt.show()

  plt.figure(figsize=(10, 5))
  fake_scores = self.sorted_input_norms * (fake_ratio)
  plt.plot(list(range(len(self.sorted_input_norms))), self.sorted_input_norms, '.', label="Input norm", alpha=0.2,zorder=10)
  plt.plot(list(range(len(self.sorted_grad_norms))), self.sorted_grad_norms, '.', label="Gradient norm", alpha=0.2)
  plt.plot(list(range(len(fake_scores))), fake_scores, '.', label="Fake scores", alpha=0.2)
  plt.xlabel('Index')
  plt.ylabel('Score')
  plt.title(f'{self.__name__}: Input norm vs gradient norm')
  plt.legend()
  plt.show()

  # Compute Spearman rank correlation from scipy
  corr = stats.spearmanr(self.input_norms, self.grad_norms)[0]
  print("Correlation between input and gradient norms (Spearman):", corr)

  corr = stats.spearmanr(fake_scores, self.sorted_input_norms)[0]
  print("Correlation between fake scores and input norms (Spearman):", corr)

  corr = stats.spearmanr(fake_scores, self.sorted_grad_norms)[0]
  print("Correlation between fake scores and gradient norms (Spearman):", corr)

#%%

TrainStats.plot()

#%%
TestStats.plot()

#%%

# Plot Gumbel(0,1) pdf
x = np.linspace(-2, 2, 100)
plt.plot(x, stats.gumbel_l.pdf(x, loc=0, scale=0.08*6**0.5/3.141592), label="Gumbel(0,1)")
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title(f'{__name__}: Gumbel(0,1) pdf')
plt.legend()
plt.show()
