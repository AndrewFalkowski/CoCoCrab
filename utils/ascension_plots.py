import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize

from .composition import _element_composition
from scipy import stats
import seaborn as sns
plt.rcParams.update({'font.size': 14})

#%%

def property_ascension_plot(optim_frac_df, prop0, prop1, errbar=False, save=False):
  
    colors = sns.color_palette('mako', 2)
    fig, ax1 = plt.subplots()
    epochs = optim_frac_df.index.values
      
    color = colors[1]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(f'{prop1}', color=color)
    if not prop1 == 'Loss':
        if errbar:
            ax1.errorbar(epochs, optim_frac_df[f'{prop1}'], 
                     yerr=optim_frac_df[f'{prop1} UNC'], 
                     color=color, mec='k', alpha=0.35, marker='s')
        ax1.plot(optim_frac_df[f'{prop1}'], color=color, mec='k', alpha=0.35, marker='s')
    else:
        ax1.plot(optim_frac_df['Loss'], color=color, mec='k', alpha=0.35, marker='s')
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
    ax2.set_ylabel(f'{prop0}', color=color)
    if errbar:
        ax2.errorbar(epochs, optim_frac_df[f'{prop0}'], 
                     yerr=optim_frac_df[f'{prop0} UNC'], 
                     color=color, mec='k', alpha=0.35, marker='o')
    else:
        ax2.plot(optim_frac_df[f'{prop0}'], color=color, mec='k', 
                 alpha=0.35, marker='o')
    
    
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


def element_ascension_plot(optim_frac_df):
    colors = sns.color_palette()
    elems = optim_frac_df.iloc[0,0]
    num_elems = int(len(optim_frac_df.iloc[0,0]))
    atom_percent = (np.concatenate(optim_frac_df['Fractions'].values)\
        .reshape(-1,num_elems))*100
    
    fig = plt.figure(figsize=(5,5))
    fig, ax1 = plt.subplots()
    for elem in range(num_elems):
        plt.plot(atom_percent[:,elem], linestyle=None, marker='s', 
                 color=colors[elem], alpha=0.35, mec='k', label = f'{elems[elem]}')
    plt.legend(loc='upper left', bbox_to_anchor=(1.04,1), framealpha=0.95)
    ax1.tick_params(direction='in', length=7,top=True, right=True)
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax1.get_xaxis().set_minor_locator(minor_locator_x)
    ax1.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    top=True)
    
    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Atomic Percent (%)')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()