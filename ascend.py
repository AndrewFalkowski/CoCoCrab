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
from utils.get_compute_device import get_compute_device
from utils.ascension_utils import *
from utils.ascension_plots import *

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

src = torch.tensor([14, 23, 31]).view(1,-1)
num_elems = int(src.shape[1])
frac = torch.ones(num_elems).view(1,-1)


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