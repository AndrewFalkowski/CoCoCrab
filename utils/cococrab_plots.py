import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm

from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import Normalize
import matplotlib.gridspec as gridspec
from utils.cococrab_utils import elem_lookup
from .composition import _element_composition
from scipy import stats
import seaborn as sns
plt.rcParams.update({'font.size': 14})

#%%

def property_optim_plot(optim_frac_df, prop0, prop1, save_dir=None):
    '''
    Parameters
    ----------
    optim_frac_df : Pandas DataFrame
        Pandas DataFrame produced by calling CoCoCrab.optimize
    prop0 : str
        Optimized property to plot
    prop1 : str, optional
        Second optimized property to plot, defaults to loss.
    save_dir : str, optional
        The directory to save produced plots to The default is None.

    Returns
    -------
    Plot of property response to changes in elemental fractions
    '''
    colors = sns.color_palette('mako', 2)
    fig, ax1 = plt.subplots()
    epochs = optim_frac_df.index.values

    color = colors[1]
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(f'{prop1}', color=color)
    if not prop1 == 'Loss':
        ax1.errorbar(epochs, optim_frac_df[f'{prop1}'],
                     yerr=optim_frac_df[f'{prop1} UNC'],
                     color=color, mec='k', alpha=0.35, marker='s')
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
    ax2.errorbar(epochs, optim_frac_df[f'{prop0}'],
                 yerr=optim_frac_df[f'{prop0} UNC'],
                 color=color, mec='k', alpha=0.35, marker='o')
    ax2.tick_params(axis='y', labelcolor=color, direction='in',
                        length=7)
    # ax2.set_ylim(120,170)
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax2.get_xaxis().set_minor_locator(minor_locator_x)
    ax2.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in', labelcolor=color,
                    length=4,
                    right=True,
                    top=True)

    plt.draw()
    if save_dir is not None:
        delim = '-'
        fig_name = f'{save_dir}/{delim.join(optim_frac_df.iloc[0,0])}_property_optimization.png'
        os.makedirs(save_dir, exist_ok=True)
        figure = plt.gcf()
        figure.set_size_inches(5,5)
        plt.savefig(fig_name, dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()


def element_optim_plot(optim_frac_df, save_dir=None):
    '''
    Parameters
    ----------
    optim_frac_df : Pandas DataFrame
        Pandas DataFrame produced by calling CoCoCrab.optimize
    save_dir : str, optional
        The directory to save produced plots to The default is None.

    Returns
    -------
    Plot of changes in atomic percent of elements during optimization

    '''
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
    plt.legend(loc='upper left', framealpha=0.95)
    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()
    ax1.tick_params(direction='in', length=7,top=True, right=True, left=True)
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax1.get_xaxis().set_minor_locator(minor_locator_x)
    ax1.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    left=True,
                    top=True)

    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Atomic Percent (%)')
    plt.draw()
    if save_dir is not None:
        delim = '-'
        fig_name = f'{save_dir}/{delim.join(optim_frac_df.iloc[0,0])}_element_fractions.png'
        os.makedirs(save_dir, exist_ok=True)
        figure = plt.gcf()
        figure.set_size_inches(5,5)
        plt.savefig(fig_name, dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()


def two_panel_optim(optim_frac_df, prop0, prop1, save_dir):
    '''
    Parameters
    ----------
    optim_frac_df : Pandas DataFrame
        Pandas DataFrame produced by calling CoCoCrab.optimize
    prop0 : str
        Optimized property to plot
    prop1 : str, optional
        Second optimized property to plot, defaults to loss.
    save_dir : str, optional
        The directory to save produced plots to The default is None.

    Returns
    -------
    Two panel plot of changes in predicted property(ies) and atomic number
    during optimization. These are the plots used in the corresponding paper.
    '''

    fig = plt.figure(figsize=(11,5))
    spec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, width_ratios=[1, 1],
                             wspace=0.35)
    ax1 = fig.add_subplot(spec[0, 0])
    plt.text(0.5, 1.1, '(a)', horizontalalignment='center',
             verticalalignment='center', transform=ax1.transAxes)
    colors = sns.color_palette('mako', 2)
    epochs = optim_frac_df.index.values

    color = colors[1]
    ax1.set_xlabel('Epoch')
    # ax1.set_ylabel(f'{prop1}', color=color)
    ax1.set_ylabel(f'{prop1}', color=color)
    if not prop1 == 'Loss':
        ax1.errorbar(epochs, optim_frac_df[f'{prop1}'],
                     yerr=optim_frac_df[f'{prop1} UNC'],
                     color=color, mec='k', alpha=0.35, marker='s')
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


    ax11 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = colors[0]
    ax11.set_ylabel(f'{prop0}', color=color)
    ax11.errorbar(epochs, optim_frac_df[f'{prop0}'],
                 yerr=optim_frac_df[f'{prop0} UNC'],
                 color=color, mec='k', alpha=0.35, marker='o')
    ax11.tick_params(axis='y', labelcolor=color, direction='in',
                        length=7)
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax11.get_xaxis().set_minor_locator(minor_locator_x)
    ax11.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in', labelcolor=color,
                    length=4,
                    right=True,
                    top=True)


    ax2 = fig.add_subplot(spec[0, 1])
    plt.text(0.5, 1.1, '(b)', horizontalalignment='center',
             verticalalignment='center', transform=ax2.transAxes)
    colors = sns.color_palette()
    elems = optim_frac_df.iloc[0,0]
    num_elems = int(len(optim_frac_df.iloc[0,0]))
    atom_percent = (np.concatenate(optim_frac_df['Fractions'].values)\
        .reshape(-1,num_elems))*100

    for elem in range(num_elems):
        plt.plot(atom_percent[:,elem], linestyle=None, marker='s',
                 color=colors[elem], alpha=0.35, mec='k', label = f'{elems[elem]}')
    plt.legend(loc='upper left', ncol=2, framealpha=0.95)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()
    ax2.tick_params(direction='in', length=7,top=True, right=True, left=True)
    minor_locator_x = AutoMinorLocator(2)
    minor_locator_y = AutoMinorLocator(2)
    ax2.get_xaxis().set_minor_locator(minor_locator_x)
    ax2.get_yaxis().set_minor_locator(minor_locator_y)
    plt.tick_params(which='minor',
                    direction='in',
                    length=4,
                    right=True,
                    left=True,
                    top=True)

    plt.ylim(0, 100)
    plt.xlabel('Epoch')
    plt.ylabel('Atomic Percent (%)')
    if save_dir is not None:
        delim = '-'
        fig_name = f'{save_dir}/{delim.join(optim_frac_df.iloc[0,0])}_jointplot.png'
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.draw()
    plt.pause(0.001)
    plt.close()

def pareto_plot(optim_frac_df, prop0, prop1, prop0_target, prop1_target):
    fig = plt.figure(figsize=(5,5))
    fig, ax = plt.subplots()
    x = optim_frac_df[f'{prop0}']
    y = optim_frac_df[f'{prop1}']
    plt.plot(x, y, linestyle='None', marker='o', mec='g', mfc='w', alpha=0.75)
    if prop0_target == 'max':
        plt.gca().invert_xaxis()
    if prop1_target == 'max':
        plt.gca().invert_yaxis()
    
    fig = plt.gcf()
    fig.set_size_inches(5,5)

    plt.xlabel(f'{prop0}')
    plt.ylabel(f'{prop1}')
    
    
    plt.show()