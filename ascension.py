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

# indicate desired properties to optimize

# prop0='aflow__agl_thermal_conductivity_300K'
prop0='aflow__ael_bulk_modulus_vrh'

prop1='aflow__agl_thermal_conductivity_300K' # Leave as loss if only want to optimize on prop0

# Input elements numbers in src vector to base composition on.
# Hydrogen is element 0

src = torch.tensor([[19,
                     13,
                     7]])

AscendModel = AscendedCrab(src, prop0, prop1)

optimized_frac_df = AscendModel.ascend(epochs=100)

property_ascension_plot(optimized_frac_df, prop0, prop1, errbar=True)

#
# dummy_src = src
# dummy_frac = torch.tensor([[0.00777465, 0.46796364, 0.52426136]])
#
# model_0 = load_model(prop0)
# model_1 = load_model(prop1)
# scale_pred_0, _, prop0_pred, prop0_unc = model_0.single_predict(dummy_src, dummy_frac)
# scale_pred_1, __, prop1_pred, prop1_unc = model_1.single_predict(dummy_src, dummy_frac)


# enter a source composition using the element numbers as they appear in
# the peridodic table

# give some recommendations

# src = [1,2,3]

# prop1 = 'bulk mod'

#defaults to loss
# prop2 = 'loss'

# call print results method

# plotting function
# simple plot
# plot with uncertainty
# mutlipanel plot
