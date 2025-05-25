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
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, LogisticRegression
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error
from itertools import product
from src.util.common import hit_ratio
import json

from datetime import datetime, timedelta
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# set user-defined functions
##################################################################################################################################################################

def _initialize_result_dataframe(CDD_METH):
    """
    Initialize result dataframe containing results of grid search

    :param  CDD_METH:   cdd method
    :return res_df_cdd: initial result dataframe
    """
    # initialize result dataframe
    cdd_param_grid  = CDD_PARAM_GRID[CDD_METH]
    cdd_param_cols  = [param.lower() for param in cdd_param_grid.keys()]
    res_df_cdd      = pd.DataFrame(columns = cdd_param_cols)

    return res_df_cdd

def _split_dataset(X, y, start_idx, end_idx):
    """
    Split dataset into training/test set

    :param  X:          input
    :param  y:          target
    :param  start_idx:  starting index of training set
    :param  end_idx:    end index of traning set
    :return X_tr, X_te, y_tr, y_te: training/test set
    """
    # divide dataset into training/test set
    X_tr, X_te = X.iloc[start_idx:end_idx, :], X.iloc[end_idx:, :] 
    y_tr, y_te = y[start_idx:end_idx],         y[end_idx:] 

    return X_tr, X_te, y_tr, y_te

def _scale_dataset(X_tr, X_te, scaler):
    """
    Scale dataset

    :param  X_tr:   input of training set
    :param  X_te:   input of test set
    :param  scaler: scaler
    :return X_tr_norm, X_te_norm: scaled input of training/test set
    """
    # scale dataset
    scl      = scaler()
    X_tr_scl = scl.fit_transform(X_tr)
    X_te_scl = scl.transform(X_te)

    return X_tr_scl, X_te_scl

def _train_ml_model(X_tr, y_tr, ML_METH):
    """
    Train ml model

    :param  X_tr:    input of training set
    :param  y_tr:    target of training set
    :param  ML_METH: ml method
    :return ml_mdl:  trained ml model
    """
    # build and train ml model
    ml_mdl = ML[ML_METH]
    ml_mdl.fit(X_tr, y_tr) 

    return ml_mdl

def _generate_parameter_combinations(CDD_METH):
    """
    Generate all hyperparameter combinations

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

def run_experiment_cdda(X, y, init_tr_start_idx, init_tr_end_idx, init_num_tr, ML_METH_LIST, CDD_METH_LIST, CDA_METH_LIST, RES_PATH):
    """
    Run cdda experiment

    :param X:                   input 
    :param y:                   target
    :param init_tr_start_idx:   initial starting index of training set
    :param init_tr_end_idx:     initial end index of training set
    :param init_num_tr:         number of data points for initial ml model training
    :param ML_METH_LIST:        list of ml methods
    :param CDD_METH_LIST:       list of cdd methods
    :param CDA_METH_LIST:       list of cda methods
    :param RES_PATH: result path
    :return
    """
    for ML_METH, CDD_METH, CDA_METH in product(ML_METH_LIST, CDD_METH_LIST, CDA_METH_LIST):
        print('*' * 50)
        print(f'ML_METH: {ML_METH}, CDD_METH: {CDD_METH}, CDA_METH: {CDA_METH}')

        # initialize result dataframe of cdd
        res_df_perf     = _initialize_result_dataframe(CDD_METH=CDD_METH)
        res_df_pred     = _initialize_result_dataframe(CDD_METH=CDD_METH)
        param_comb_list = _generate_parameter_combinations(CDD_METH=CDD_METH)
        num_param_comb  = len(param_comb_list)
        for idx in range(num_param_comb):
            # check start time
            start_time  = time.time()     
            
            # print progress
            print(f'{CDD_METH}, {idx+1}/{num_param_comb}, {param_comb_list[idx]}')

            # set idx-th parameter combination
            param_comb_idx = param_comb_list[idx]

            # split and normalize dataset 
            X_tr, X_te, y_tr, y_te = _split_dataset(X=X, y=y, start_idx=init_tr_start_idx, end_idx=init_tr_end_idx)
            X_tr_scl, X_te_scl     = _scale_dataset(X_tr=X_tr, X_te=X_te, scaler=StandardScaler) # MinMaxScaler 

            # train initial ml model
            ml_mdl  = _train_ml_model(X_tr=X_tr_scl, y_tr=y_tr, ML_METH=ML_METH)

            # build cdd model
            cdd_mdl = CDD[CDD_METH](**param_comb_idx)
            
            adapt_end_idx, det_end_idx = init_tr_end_idx, len(y)
            while adapt_end_idx < det_end_idx:
                # detect drift
                res_det = cdd_mdl.detect_drift(ml_mdl = ml_mdl, X_te = X_te_scl, y_te = y_te)

                # build cda model
                cda_mdl = CDA[CDA_METH](det_mdl = cdd_mdl, res_det = res_det, init_tr_end_idx = init_tr_end_idx)

                # set adpatation period
                adapt_start_idx, adapt_end_idx = cda_mdl.set_adaptation_period()

                if adapt_end_idx >= det_end_idx:
                    break 
                #end if

                # split and scale dataset 
                X_tr, X_te, y_tr, y_te = _split_dataset(X=X, y=y, start_idx=adapt_start_idx, end_idx=adapt_end_idx)
                X_tr_scl, X_te_scl     = _scale_dataset(X_tr=X_tr, X_te=X_te, scaler=StandardScaler)

                # retrain ML model
                ml_mdl  = _train_ml_model(X_tr=X_tr_scl, y_tr=y_tr, ML_METH=ML_METH)
            # end while

            end_time = time.time() # check end time
            
            res_perf_idx = {
                'adapt_prd': cdd_mdl.adapt_prd_list, 

                # 241227: accuacy_score는 분류에 국한되어있으므로, hit ratio등을 반영할 수 있게
                # 예측 결과(1과 0이 담긴) 벡터를 불러와 sum/len 형식으로 변환이 필요함
                'acc':      np.round(accuracy_score(cdd_mdl.y_pred_te, cdd_mdl.y_real_te), 2),
                'exec_time': np.round(end_time - start_time, 2)
            }

            res_pred_idx = {
                # 추가로 예측 결과(1과 0이 담긴) 벡터를 불러와야함
                'y_real': cdd_mdl.y_real_te,
                'y_pred': cdd_mdl.y_pred_te
            }

            # set result dataframe
            res_df_perf_idx  = pd.DataFrame([{**param_comb_idx, **res_perf_idx}])
            res_df_pred_idx = pd.DataFrame({**param_comb_idx, **res_pred_idx})

            res_df_perf  = pd.concat([res_df_perf, res_df_perf_idx], ignore_index = True)
            res_df_pred  = pd.concat([res_df_pred, res_df_pred_idx], ignore_index = True)
        # end for
        
        print(res_df_perf)
        print('*' * 200)

        # save result dataframe into csv format
        res_df_perf_path = RES_PATH['PERF_ROOT'] + RES_PATH['CDDA_DHY']
        res_df_pred_path = RES_PATH['PRED_ROOT'] + RES_PATH['CDDA_DHY']

        res_df_perf_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_PERF_{VER}.csv'
        res_df_pred_name = f'{DATE}_{DATA_TYPE}_{DATA}_{ML_METH}_{CDD_METH}_{CDA_METH}_PRED_{VER}.csv'
        
        res_df_perf.to_csv(res_df_perf_path + res_df_perf_name)
        res_df_pred.to_csv(res_df_pred_path + res_df_pred_name)
    # end for

    return None