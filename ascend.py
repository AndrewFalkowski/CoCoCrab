#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:19:11 2021

@author: andrewf
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from tabulate import tabulate

from crabnet.neokingcrab import CrabNet
from crabnet.neomodel import Model
# from utils.utils import EDMDataset
from utils.get_compute_device import get_compute_device

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32

#%%

mat_prop = 'aflow__ael_bulk_modulus_vrh'
data_dir = 'data/materials_data'
mat_prop = 'aflow__ael_bulk_modulus_vrh'
verbose = False
file_name = 'test.csv'

data = rf'{data_dir}/{mat_prop}/{file_name}'

bulk_name = 'aflow__ael_bulk_modulus_vrh'
bulk_model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{bulk_name}', verbose=False)
bulk_model.load_network(f'{bulk_name}.pth')
bulk_model.load_data(data, batch_size=2**9, train=False)

energy_name = 'decomposition_energy'
energy_model = Model(CrabNet(compute_device=compute_device).to(compute_device),
                  model_name=f'{energy_name}', verbose=False)
energy_model.load_network(f'{energy_name}.pth')
energy_model.load_data(data, batch_size=2**9, train=False)

#%%

def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps/masked_sums)

class EDMDataset(Dataset):
    """
    Get X and y from EDM dataset.
    """

    def __init__(self, dataset, n_comp):
        self.data = dataset
        self.n_comp = n_comp

        self.X = self.data[0]
        self.y = self.data[1]
        # self.formula = torch.tensor(self.data[2])
        self.formula = torch.tensor([0.0])

        self.shape = [(self.X.shape), (self.y.shape), (self.formula.shape)]

    def __str__(self):
        string = f'EDMDataset with X.shape {self.X.shape}'
        return string

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        X = self.X[idx, :, :]
        y = self.y[idx]
        formula = self.formula[idx]

        X = torch.as_tensor(X, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)
        # X = torch.as_tensor(X, dtype=torch.float)
        # y = torch.as_tensor(y, dtype=torch.float)

        return (X, y, formula)

def ascension_loader(src, frac, batch_size=2**9, pin_memory=False):
    EDM = torch.cat((src, frac), 0).view(1, -1, 1)
    n_elements = len(src)
    ascension_data = (EDM, dummy_y, dummy_form)
    ascension_dataset = EDMDataset(ascension_data, n_elements)
    data_loader = DataLoader(ascension_dataset,
                             batch_size = batch_size,
                             pin_memory = pin_memory)
    return data_loader

def ascension_plot(losses, bulk_mods, bunc, epoch):
  
  colors = sns.color_palette('mako', 2)
  fig, ax1 = plt.subplots()
  epochs = np.arange(epoch)

  color = colors[1]
  ax1.set_xlabel('Epoch')
  ax1.set_ylabel('Loss', color=color)
  ax1.plot(losses, color=color, mec='k', alpha=0.35, marker='s')
  ax1.tick_params(axis='y', labelcolor=color)
  ax1.tick_params(direction='in',
                      length=7,top=True, right=True)

  minor_locator_x = AutoMinorLocator(2)
  minor_locator_y = AutoMinorLocator(2)
  ax1.get_xaxis().set_minor_locator(minor_locator_x)
  ax1.get_yaxis().set_minor_locator(minor_locator_y)
  plt.tick_params(which='minor',
                  direction='in', labelcolor=color,
                  length=4,
                  right=True,
                  top=True)



  ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

  color = colors[0]
  ax2.set_ylabel('Bulk Modulus', color=color)  # we already handled the x-label with ax1
  # ax2.plot(bulk_mods, color=color, mec='k', alpha=0.35, marker='o')
  ax2.errorbar(epochs, bulk_mods, yerr=bunc, color=color, mec='k', alpha=0.35, marker='o')
  ax2.tick_params(axis='y', labelcolor=color, direction='in',
                      length=7)

  minor_locator_x = AutoMinorLocator(2)
  minor_locator_y = AutoMinorLocator(2)
  ax2.get_xaxis().set_minor_locator(minor_locator_x)
  ax2.get_yaxis().set_minor_locator(minor_locator_y)
  plt.tick_params(which='minor',
                  direction='in', labelcolor=color,
                  length=4,
                  right=True,
                  top=True)

  fig.tight_layout()  # otherwise the right y-label is slightly clipped
  plt.show()

def elem_lookup(src):
    try:
        numpy_src = src.numpy().reshape(-1)
    except:
        numpy_src = src
    all_symbols = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na',
                   'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc',
                   'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                   'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb',
                   'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb',
                   'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm',
                   'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
                   'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl',
                   'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa',
                   'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md',
                   'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg',
                   'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
    
    elem_names = [all_symbols[i] for i in numpy_src]
    return elem_names

#%%

src = torch.tensor([73, 6]).view(1,-1)
num_elems = int(src.shape[1])
frac = torch.rand(num_elems).view(1,-1)
# dummy_y = torch.tensor([100.00])
# dummy_form = ['Hg1Al2']


epochs = 100

delim ='-'
print(f'\n\nOptimizing {delim.join(elem_lookup(src))} System in {epochs} Epochs... \n'.title())

losses = []
srcs = []
fracs = []
bulk_mods = []
bulk_mod_uncs = []
d_nrgs = []
d__nrg_uncs = []


src = src.to(compute_device,
        dtype=torch.long,
        non_blocking=True)

frac_mask = torch.where(src != 0, 1, 0)
frac_mask = frac_mask.to(compute_device, 
                      dtype=torch.float,
                      non_blocking=True)

frac = frac.to(compute_device,
        dtype=torch.float,
        non_blocking=True)


optim_lr = 0.025
optimizer = optim.Adam([frac.requires_grad_()], lr=optim_lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, gamma=0.1, last_epoch=-1, verbose=False)
criterion = nn.L1Loss()
criterion = criterion.to(compute_device)


for epoch in tqdm(range(epochs)):
    # print(f'frac: {frac}')
    soft_frac = masked_softmax(frac, frac_mask)
    
    optimizer.zero_grad()
    bpred, buncertainty = bulk_model.predict(src, soft_frac)
    
    # factor = torch.tensor(bpred+epred).to(compute_device)
    loss = criterion(bpred, torch.tensor([[100000.0]]).to(compute_device))
    # loss = loss + criterion(epred, torch.tensor([10000.0]).to(compute_device))
    
    loss.backward()
    optimizer.step()
    scheduler.step()

    losses.append(loss.item())
    srcs.append(src)
    fracs.append(soft_frac)
    bulk_mods.append(bpred.item())
    bulk_mod_uncs.append(buncertainty.item())
        # d_nrgs.append(epred)
        # d__nrg_uncs.append(euncertainty)

srcs = [elem_lookup(item.detach().numpy().reshape(-1)) for item in srcs]
fracs = [item.detach().numpy().reshape(-1) for item in fracs]

optimized_frac_df = pd.DataFrame(
    {'Elements': srcs,
     'Fractions': fracs,
     'P1': bulk_mods,
     'P1 Uncertainty': bulk_mod_uncs,
     'Loss': losses
    })

print('\n-----------------------------------------------------')
print('\nOptimized Fractional Composition:\n'.title())

# display(all_df.tail(1))
print(optimized_frac_df.tail(1).iloc[:,0:2].to_markdown(index=False, tablefmt="simple"))
print('\n')
print(optimized_frac_df.tail(1).iloc[:,2:].to_markdown(index=False, tablefmt="rst"))

# print(tabulate([['Alice', 24], ['Bob', 19]], headers=['Element', 'Fraction']))

#%%
ascension_plot(losses, bulk_mods, bulk_mod_uncs, epochs)