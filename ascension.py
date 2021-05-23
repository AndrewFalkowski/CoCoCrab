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
from crabnet.model import Model
# from utils.utils import EDMDataset
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

# indicate the desired properties to optimize
# listed properties should correspond names of trained models in directories

prop0='aflow__ael_bulk_modulus_vrh'
prop0_target = 'max' # whether to maximize or minimize this property
prop1='Loss' # Set to 'Loss' for single optim
prop1_target = 'min'

# The src tensor contains the atomic numbers of the elements whose fractions 
# will be optimized. Elements should be listed according to their atomic number

# The Ti-Cd-C system would be listed as [[22,48,6]]

# src = torch.tensor([[73, 74, 75, 76, 77, 42, 43, 5, 6, 7]])
src = torch.tensor([[20,13,14,8]])
# src = torch.tensor([[22,48,6]])

# AscendModel initializes the models and variables for gradient ascent
AscendModel = AscendedCrab(src, prop0, prop1, prop0_target, prop1_target, 
                            alpha=0.8, lr=0.05)

# The ascend model conducts the gradient ascent and returns a df with results
optimized_frac_df = AscendModel.ascend(epochs=100)

# plot the changes in predicted properties across epochs during gradient ascent
property_ascension_plot(optimized_frac_df, prop0, prop1, save_dir=None)

# show changes in atomic percent for candidate elements during gradient ascent
element_ascension_plot(optimized_frac_df, save_dir=None)

# save gradient ascent results to csv file
# save_results(optimized_frac_df, save_dir='results')
