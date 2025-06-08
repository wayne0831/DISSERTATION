##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Experiment
# SECTION: main_real
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import json
from src.common.config import *
from src.util.experiment import *
from src.util.preprocess import *
from src.util.metric import hit_ratio
from itertools import product

import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

##################################################################################################################################################################
# version control
##################################################################################################################################################################

DATE            = '250609'
DATA_TYPE       = 'ART'   # 'APP', 'ART', 'REAL'
DATA            = 'LED'   # 'POSCO', 'PDX'
PROB_TYPE       = 'CLF'   # 'CLF', 'REG'
ML_METH_LIST    = ['NB'] # LASSO, LOG_REG

"""
250311
Completed: {
    SingleWindow:    ['DDM', 'FHDDM', 'MDDM-a/g/e', 'BDDM']
    MultipleWindows: ['ADWIN', 'STEPD', 'WSTD', 'FDD-t/s/p', 'FHDDMS']
        TBD: PL, AL
}

Ongoing: ['DOER']
"""
# CDD_METH_LIST   = ['DDM', 'FHDDM', 'MDDM', 'BDDM', 'ADWIN', 'STEPD', 'WSTD', 'FDD', 'FHDDMS']
CDD_METH_LIST   = ['DDM']
CDA_METH_LIST   = ['REC']
VER             = 'V4'    # PDX -> v1: n_m_m[:39] / v2: n_m_m[:19] / v3: posco

#####################################################cls#############################################################################################################
# load and preprocess dataset
##################################################################################################################################################################

# divide dataset into X and y
X, y = set_synthethic_dataset(data=SYN_DATA)

print(X)
print(y)

##################################################################################################################################################################
# set experiment setting
##################################################################################################################################################################

run_experiment_cdda(X=X, 
                    y=y, 
                    scaler=OnlineStandardScaler(), 
                    tr_start_idx=0, 
                    tr_end_idx=1, 
                    len_batch=1, 
                    min_len_tr=1, 
                    perf_bnd=None,
                    prob_type='CLF',
                    res_df_perf_path=RES_PATH['PERF_ROOT'] + RES_PATH['CDDA'], 
                    res_df_pred_path=RES_PATH['PRED_ROOT'] + RES_PATH['CDDA'])