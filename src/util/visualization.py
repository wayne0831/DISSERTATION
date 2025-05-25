##################################################################################################################################################################
# PROJECT: TM_MLOps
# CHAPTER: Util
# SECTION: Experiment
# AUTHOR: Yang et al.
# DATE: since 24.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from src.common.config import *
from src.util import *
from algorithm.ml.ML import *

import pandas as pd
import numpy as np
import time
import os

import matplotlib.pyplot as plt

##################################################################################################################################################################
# set user-defined functions
##################################################################################################################################################################


def plot_performances(train_len : int, in_sample_error : float, *args, colors = None, legends = None, title : str):
    if colors is None:
        colors = plt.cm.get_cmap('tab10', len(args))
    
    if legends is None:
        legends = [f'List {i+1}' for i in range(len(args))]
    
    in_sample_error_ls = [in_sample_error] * train_len
    
    fig, ax = plt.subplots()
    ax.set_xlabel('Data index')
    ax.plot(list(range(1, train_len + 1)), in_sample_error_ls, 'g--')
    for i, y_values in enumerate(args):
        color = colors(i) if callable(colors) else colors[i % len(colors)]
        ax.plot(list(range(train_len + 1, train_len + len(y_values) + 1)),
                y_values, color=color, label = legends)
    
    plt.legend(legends)
    plt.title(title)
    plt.show()