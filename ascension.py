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

# indicate desired properties to optimize from list of trained models

prop0='aflow__ael_bulk_modulus_vrh'

prop1='aflow__agl_thermal_conductivity_300K' # Leave as 'Loss' if only want to optimize prop0

# Input elements numbers in src vector to base composition on.
# Hydrogen is considered element 0 in this representation

src = torch.tensor([[19,
                     13,
                     12,
                     7]])

alpha = 0.5 # weight parameter for optimization, applies directly to prop0

AscendModel = AscendedCrab(src, prop0, prop1, alpha, lr=0.025)

optimized_frac_df = AscendModel.ascend(epochs=100)

property_ascension_plot(optimized_frac_df, prop0, prop1, errbar=True)

element_ascension_plot(optimized_frac_df)

#
# dummy_src = src
# dummy_frac = torch.tensor([[0.85224, 0.0497999, 0.04282987, 0.05512916]])

# model_0 = load_model(prop0)
# model_1 = load_model(prop1)
# sp0, _, prop0_pred, prop0_unc = model_0.single_predict(dummy_src, dummy_frac)
psp1, _, rop1_pred, prop1_unc = model_1.single_predict(dummy_src, dummy_frac)


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
