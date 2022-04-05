#%%
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

from crabnet.cococrab import CrabNet
from crabnet.model import Model
from utils.get_compute_device import get_compute_device
from utils.cococrab_utils import *
from utils.cococrab_plots import *

import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

compute_device = get_compute_device(prefer_last=True)
RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)
data_type_torch = torch.float32
#%%

# Indicate the desired properties to optimize

# Listed properties should correspond trained models in models/trained_models

prop0 = "aflow__ael_bulk_modulus_vrh"
prop0_target = "max"  # whether to maximize or minimize this property
prop1 = "Loss"  # Set to 'Loss' for single property optimization
prop1_target = "min"

# Alpha governs the balancing of loss functions in multi-property optimization
# tasks. Higher alpha prioritizes prop0, lower alpha prioritizes prop1, when
# alpha=0.5, neither is prioritized
alpha = 0.5

# The src tensor contains the atomic numbers of the elements whose fractions
# will be optimized. Elements should be listed according to their atomic number

# The Ti-Cd-C system would be listed as [[22,48,6]]
src = torch.tensor([[22, 48, 6]])


# AscendModel initializes the models and variables for optimization
OptimModel = CoCoCrab(
    src,
    prop0,
    prop1,
    prop0_target,
    prop1_target,
    alpha,
    lr=0.05,
    dopant_threshold=0.001,  # set molar fraction threshold for dopant inclusion
)

# The ascend model conducts the gradient ascent and returns a df with results
optimized_frac_df = OptimModel.optimize_comp(epochs=100)

# plot the changes in predicted properties across epochs during gradient ascent
# property_optim_plot(optimized_frac_df, prop0, prop1, save_dir="figures")

# show changes in atomic percent for candidate elements during gradient ascent
# element_optim_plot(optimized_frac_df, save_dir=None)

# show both plots on two panel plot, used to produce paper figures
two_panel_optim(optimized_frac_df, prop0, prop1, save_dir="figures")

if prop1 != "Loss":
    pareto_plot(optimized_frac_df, prop0, prop1, prop0_target, prop1_target)

# save gradient ascent results to csv file
save_results(optimized_frac_df, save_dir="results")

# %%
