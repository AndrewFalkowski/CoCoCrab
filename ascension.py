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
prop1='aflow__agl_thermal_conductivity_300K' # Leave as 'Loss' for single optim

# The src tensor contains the atomic numbers of the elements whose fractions 
# will be optimized. Hydrogen is considered element 0 in this representation.

# The Ti-Cd-C system would be listed as [[21,47,5]]

src = torch.tensor([[19,
                     13,
                     12,
                     7]])

# AscendModel initializes the models and variables for gradient ascent
AscendModel = AscendedCrab(src, prop0, prop1, alpha=0.5, lr=0.1)

# The ascend model conducts the gradient ascent and returns a df with results
optimized_frac_df = AscendModel.ascend(epochs=100)

# plot the changes in predicted properties across epochs during gradient ascent
property_ascension_plot(optimized_frac_df, prop0, prop1, save_dir='figures',
                        errbar=True)

# show changes in atomic percent for candidate elements during gradient ascent
element_ascension_plot(optimized_frac_df, save_dir='figures')

# save gradient ascent results to csv file
save_results(optimized_frac_df, save_dir='results')

