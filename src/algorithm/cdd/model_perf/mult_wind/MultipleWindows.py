##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Algorithm
# SECTION: Concept Drift Detection
# SUB-SECTION: Model Performance-based
# SUB-SUBSECTION: Single Window-based
# AUTHOR: Yang et al.
# DATE: since 25.05.26
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

import pandas as pd
import numpy as np
import math
import os

from src.util.train import set_ml_dataset, run_ml_model
from src.util.preprocess import OnlineStandardScaler
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
from statsmodels.stats.proportion import proportions_ztest
from scipy.stats import ks_2samp, dirichlet, norm, mannwhitneyu
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.preprocessing import StandardScaler
#from src.util.util import run_ml_model_pipeline, calculate_attention_matrix
from datetime import datetime, timedelta

import matplotlib.pyplot as pipet
import matplotlib.patches as patches
import itertools
import time
import warnings
warnings.filterwarnings('ignore')

##################################################################################################################################################################
# Multiple Window-based Methods
##################################################################################################################################################################

##################################################################################################################################################################
# MR-DDM (Multi-Resolution Drift Detection Method, 2025) 
##################################################################################################################################################################

class MRDDM:
    def __init__(self, **kwargs):
        # hyperparameters of MRDDM
        self.alpha_d  = kwargs['alpha_d']
        self.len_step = kwargs['len_step'] # length of step
        self.len_sw   = kwargs['len_sw'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        #self.warn_prd     = []
        self.res_pred_tmp = []

        # cumulative data points, X_cum and y_cum
        # Note) They are used to train the ml model, when the model does not support partial_fit
        self.X_cum = None
        self.y_cum = None

        # dict containing results of cdda
        self.res_cdda = {
            'time_idx':      [], # data index
            'y_real_list':   [], # real targets
            'y_pred_list':   [], # predictions
            'res_pred_list': [], # prediction results (= 0 or 1)
            'cd_idx':        [], # index of concept drfit
            'len_adapt':     [], # length of adaptation period
        }

    @staticmethod
    def _detect_drift(res_pred_lw, res_pred_sw, alpha_d):
        # initialize temporary state
        state = 0  # 0: stable, 1: warning, 2: drift

        num_obs    = len(res_pred_lw)
        num_lw_cor = sum(res_pred_lw)
        num_sw_cor = sum(res_pred_sw)

        num_cor_arr = np.array([num_lw_cor, num_sw_cor])
        num_obs_arr = np.array([num_obs, num_obs])

        _, p_val = proportions_ztest(num_cor_arr, num_obs_arr, alternative='two-sided')

        state = 2 if p_val < alpha_d else 0

        return state
    
    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_start_idx, te_end_idx):
        if state == 0:  # stable
            tr_start_idx = te_start_idx
            tr_end_idx   = te_end_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')    

            # set drift index
            drift_idx = te_end_idx
            
            tr_start_idx = drift_idx - min_len_tr
            tr_end_idx   = te_end_idx

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.state        = 0
            self.X_cum        = None
            self.y_cum        = None
            self.res_pred_tmp = []
        # end if

        return tr_start_idx, tr_end_idx
    
    def _reset_parameters(self):

        return None
    
    def run_cdda(self, X, y, scaler, prob_type, ml_mdl, tr_start_idx, tr_end_idx, len_batch, min_len_tr, perf_bnd):
        """
        Run Concept Drift Detection and Adaptation

        :param X:               input       (type: np.array)
        :param y:               target      (type: np.array)
        :param scaler           scaler      (type: TransformerMixin)
        :param prob_type:       problem type (clf: classification / reg: regression) (type: str)
        :param ml_mdl:          ml model   (type: str)
        :param tr_start_idx:    initial start index of training set (type: int)
        :param tr_end_idx:      initial end index of training set   (type: int)        
        :param len_batch        length of batch                     (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param perf_bnd         performance bound for treating prediction results as (in)correct (type: float)
        :return
        """
        # run process
        num_data = len(X)
        while tr_end_idx < num_data:
            # set test set index
            te_start_idx = tr_end_idx
            te_end_idx   = min(tr_end_idx + len_batch, len(X))

            print(f'tr_start_idx: {tr_start_idx} / tr_end_idx: {tr_end_idx} / te_start_idx: {te_start_idx} / te_end_idx: {te_end_idx}')

            # set dataset for ml model
            X_tr, y_tr, X_te, y_te = set_ml_dataset(tr_start_idx = tr_start_idx, 
                                                    tr_end_idx   = tr_end_idx, 
                                                    te_start_idx = te_start_idx, 
                                                    te_end_idx   = te_end_idx,
                                                    X            = X, 
                                                    y            = y)
            # cumulate incoming data points
            self.X_cum = np.concatenate([self.X_cum, X_tr]) if self.X_cum is not None else X_tr
            self.y_cum = np.concatenate([self.y_cum, y_tr]) if self.y_cum is not None else y_tr

            # train the ml model and predict the test set
            y_pred_te, res_pred_idx = run_ml_model(X_cum        = self.X_cum, 
                                                   y_cum        = self.y_cum, 
                                                   X_tr         = X_tr, 
                                                   y_tr         = y_tr, 
                                                   X_te         = X_te, 
                                                   y_te         = y_te,
                                                   y            = y,
                                                   scaler       = scaler, 
                                                   ml_mdl       = ml_mdl, 
                                                   prob_type    = prob_type, 
                                                   perf_bnd     = perf_bnd)

            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(X_te.index)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.len_step*self.len_sw:
                # set windows of older/recent prediction results
                res_pred_lw = self.res_pred_tmp[::self.len_step]                
                res_pred_sw = self.res_pred_tmp[-self.len_sw:]
                
                self.state = self._detect_drift(res_pred_lw = res_pred_lw, 
                                                res_pred_sw = res_pred_sw, 
                                                alpha_d     = self.alpha_d)
            # end if

            # set the start/end index of updated training set
            tr_start_idx, tr_end_idx = self._adapt_drift(state        = self.state, 
                                                         min_len_tr   = min_len_tr,
                                                         tr_start_idx = tr_start_idx, 
                                                         tr_end_idx   = tr_end_idx,
                                                         te_start_idx = te_start_idx, 
                                                         te_end_idx   = te_end_idx)
        # end while

        return None