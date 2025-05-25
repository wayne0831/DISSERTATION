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

# from src.algorithm.cdda.model_perf.sing_wind.SingleWindow import *
# from src.algorithm.cdda.model_perf.mult_wind.MultipleWindows import *
import json
from src.common.config import *
from src.util.util import hit_ratio
from itertools import product

from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

##################################################################################################################################################################
# set user-defined functions
##################################################################################################################################################################

def generate_parameter_combinations(CDD_PARAM_GRID, CDD_METH):
    """
    Generate all hyperparameter combinations

    :param  CDD_PARAM_GRID: param grid of cdd method
    :param  CDD_METH: cdd method
    :return param_comb_list: list of hyperparameter combinations
    """
    # generate combinations of hyperparameter
    cdd_params  = list(CDD_PARAM_GRID[CDD_METH].keys())    # ex) ['alpha_w', 'alpha_d'...]
    cdd_values  = list(CDD_PARAM_GRID[CDD_METH].values())  # ex) [[1.5, 2], [2.5, 3], ...]
    param_comb  = product(*cdd_values)                     # ex) [[1.5 , 2.5], [1.5, 3], ...]

    param_comb_list = []
    for comb in param_comb:
        param = dict(zip(cdd_params, comb)) # ex) {'alpha_w': 1.5, 'alpha_d': 2.5, ..}
        param_comb_list.append(param)       # ex) [{'alpha_w': 1.5, 'alpha_d': 2.5, ..}, {...}, ...]
    # end for

    return param_comb_list

def run_experiment_cdda(X, y, scaler, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd,
                        res_df_perf_path, res_df_pred_path):
    
    for ML_METH, CDD_METH, CDA_METH in product(ML_METH_LIST, CDD_METH_LIST, CDA_METH_LIST):
        print('*' * 150)
        print(f'ML_METH: {ML_METH}, CDD_METH: {CDD_METH}, CDA_METH: {CDA_METH}')

        res_df_perf = pd.DataFrame()
        res_df_pred = pd.DataFrame()
    
        param_comb_list = generate_parameter_combinations(CDD_PARAM_GRID=CDD_PARAM_GRID, CDD_METH=CDD_METH)

        for param_comb in param_comb_list:
            print(f'param_comb: {param_comb}')
            start_time = time.time()

            cdda = CDD[CDD_METH](**param_comb)
            cdda.run_cdda(X             = X, 
                          y             = y, 
                          scaler        = scaler, 
                          prob_type     = PROB_TYPE, 
                          ml_mdl        = ML[PROB_TYPE][ML_METH],
                          tr_start_idx  = tr_start_idx,
                          tr_end_idx    = tr_end_idx,
                          len_batch     = len_batch,
                          min_len_tr    = min_len_tr,
                          perf_bnd      = perf_bnd)
            
            end_time = time.time()

            time_idx    = cdda.res_cdda['time_idx']
            y_real_list = cdda.res_cdda['y_real_list']
            y_pred_list = cdda.res_cdda['y_pred_list']

            if PROB_TYPE == 'REG':
                res_perf_idx = {
                    'cdd_method':      str(CDD_METH),
                    'param':           json.dumps(param_comb),
                    'init_tr_end_idx': tr_end_idx,
                    'cd_idx':    cdda.res_cdda['cd_idx'], 
                    'num_cd':    len(cdda.res_cdda['cd_idx']),
                    'len_adapt': cdda.res_cdda['len_adapt'],
                    'mape':      np.round(mean_absolute_percentage_error(y_real_list, y_pred_list) * 100, 4),
                    'mae':       np.round(mean_absolute_error(y_real_list, y_pred_list), 4),
                    'rmse':      np.round(root_mean_squared_error(y_real_list, y_pred_list), 4),
                    'r2':        np.round(r2_score(y_real_list, y_pred_list), 4),
                    'ctq':       np.round(hit_ratio(y_real_list, y_pred_list, perf_bnd) * 100, 2),
                    'exec_time': np.round((end_time - start_time)/60, 2)
                }
            elif PROB_TYPE == 'CLF':
                res_perf_idx = {
                    'cdd_method':      str(CDD_METH),
                    'param':           json.dumps(param_comb),
                    'init_tr_end_idx': tr_end_idx,
                    'cd_idx':    cdda.res_cdda['cd_idx'], 
                    'num_cd':    len(cdda.res_cdda['cd_idx']),
                    'len_adapt': cdda.res_cdda['len_adapt'],
                    'acc':       np.round(accuracy_score(y_real_list, y_pred_list) * 100, 2),
                }                
            
            print(res_perf_idx)

            res_pred_idx = {
                'cdd_method':   str(CDD_METH),
                'param':    json.dumps(param_comb),                
                'time_idx': time_idx,
                'y_real':   y_real_list,
                'y_pred':   y_pred_list,
            }

            res_df_perf  = pd.concat([res_df_perf, pd.DataFrame([res_perf_idx])], ignore_index = True)
            res_df_pred  = pd.concat([res_df_pred, pd.DataFrame(res_pred_idx)], ignore_index = True)
        # end for param_comb

        # set prediction result dataframe
        res_df_pred_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_{VER}_PRED.csv'
        res_df_pred.to_csv(res_df_pred_path + res_df_pred_name)

        # set performance results
        res_df_perf_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_{VER}_PERF.csv'
        res_df_perf.to_csv(res_df_perf_path + res_df_perf_name)
    
    # end for param_comb ML_METH, CDD_METH, CDA_METH

    return None

##################################################################################################################################################################
# load and preprocess dataset
##################################################################################################################################################################

# load dataset
df_path = DATA_PATH[DATA_TYPE][DATA]
index   = DATASET[DATA_TYPE][DATA]['INDEX']
df      = pd.read_csv(df_path, index_col = index)

# divide dataset into X and y
input = DATASET[DATA_TYPE][DATA][f'INPUT_{VER}']
trgt  = DATASET[DATA_TYPE][DATA]['TRGT']
X = df.loc[:, input]
y = df.loc[:, trgt]
y = y.values.ravel() # Note) y must be array!

##################################################################################################################################################################
# set experiment setting
##################################################################################################################################################################

# set initial index of training set
init_tr_start_idx   = INFRM_ADAPT[DATA_TYPE][DATA]['INIT_TR_START_IDX']
init_num_tr         = INFRM_ADAPT[DATA_TYPE][DATA]['INIT_NUM_TR']
init_tr_end_idx     = init_tr_start_idx + init_num_tr

# set problem type
prob_type = DATASET[DATA_TYPE][DATA]['PROB_TYPE']

# set performance bound for determining the prediction results as right or wrong
ctq_thr = np.std(y[init_tr_end_idx:])
ctq_thr

run_experiment_cdda(X=X, 
                    y=y, 
                    scaler=None, 
                    tr_start_idx=init_tr_start_idx, 
                    tr_end_idx=init_tr_end_idx, 
                    len_batch=12, 
                    min_len_tr=1, 
                    perf_bnd=ctq_thr,
                    res_df_perf_path=RES_PATH['PERF_ROOT'] + RES_PATH['CDDA_DHY'], 
                    res_df_pred_path=RES_PATH['PRED_ROOT'] + RES_PATH['CDDA_DHY'])