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

from src.util.train import run_ml_model_pipeline
from src.util.preprocess import OnlineStandardScaler
from sklearn.preprocessing import StandardScaler

from scipy.spatial.distance import jensenshannon
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats as stats
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
# Single Window-based Methods
##################################################################################################################################################################

##################################################################################################################################################################
# DDM (Drift Detection Method, 2004) 
##################################################################################################################################################################

class DDM:
    def __init__(self, **kwargs):
        """
        Initialize values
        **kwargs: {alpha_w: float, alpha_d: float, clt_idx: int}

        :param alpha_w:     warning conf.level                    (rng: >= 1)   (type: float)            
        :param alpha_d:     drift conf.level                      (rng: >= 1)   (type: float)
        :param clt_idx:     min. num. of data points to obey CLT  (rng: >= 30)  (type: int)
        """
        # values to be reset after drift detection
        self.p_min = np.inf   # min. err. rate of the ML model
        self.s_min = np.inf   # min. std. dev. of the ML model

        # hyperparameters of DDM
        self.alpha_w = kwargs['alpha_w']
        self.alpha_d = kwargs['alpha_d'] 
        self.clt_idx = kwargs['clt_idx'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.warn_prd     = []
        self.res_pred_tmp = []
        
        #
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
    def _detect_drift(p_idx, s_idx, p_min, s_min, alpha_w, alpha_d):
        """
        Evaluate the state of the concept

        :param p_idx:       err. rate of ML model untill idx-th point   (type: float)  
        :param s_idx        std. dev. of ML model untill idx-th point   (type: float)
        :param p_min:       min. err. rate of ML model observed so far  (type: float)
        :param s_min        min. std. dev. of ML model observed so far  (type: float)
        :param alpha_w:     warning conf.level     (rng: >= 1)          (type: float)
        :param alpha_d:     drift conf.level       (rng: >= 1)          (type: float)
        :return:            state of the concept
        """
        state = 0 # 0: stable, 1: warning, 2: drift

        # compute test statistics for drift detection
        test_stat = p_idx + s_idx

        # evaluate state
        ### 250109 DHY: if 문에 >= 대신 > 사용
        ### 모든 예측이 정답(p_idx와 s_idx 모두 0)이면 p_min과 s_min도 0이됨 => state는 2가 되고 cd가 컨펌됨 
        ### 모든 예측이 정답인데 cd가 컨펌되서는 안되므로 코드상에서는 > 를 사용하고자 함        
        state = 2 if test_stat > (p_min + alpha_d * s_min) else \
                1 if test_stat > (p_min + alpha_w * s_min) else \
                0

        return state

    def _adapt_drift(self, state, min_len_tr, tr_start_idx, tr_end_idx, te_end_idx):
        """
        Adjust the training set for ml model update
        
        :param state:           state of the concept                 (type: int)
        :param min_len_tr       minimum length of training set for ml model update  (type: int)
        :param tr_start_idx:    previous start index of training set (type: int)
        :param tr_end_idx:      previous end index of training set   (type: int)        
        :param te_end_idx:      previous end index of test set       (type: int)                
        :return:                updated start index of training set  (type: int)
        """
        if state == 0:  # stable
            tr_start_idx = tr_start_idx # TODO: online learning이라서 tr_start_idx 수정해야할듯
        elif state == 1:  # warning
            # increment adaptation period
            self.warn_prd.append(te_end_idx)
            tr_start_idx = tr_start_idx
        elif state == 2: # drift
            print(f'Drift detected at {te_end_idx}')

            # set drift index
            drift_idx = te_end_idx

            # increment adaptation period
            self.warn_prd.append(drift_idx) 

            # set the start index of model update
            if (drift_idx - self.warn_prd[0]) <= min_len_tr:
                tr_start_idx = drift_idx - min_len_tr 
            else: 
                tr_start_idx = self.warn_prd[0]
            # end if

            # set the results of cdda
            self.res_cdda['cd_idx'].append(drift_idx)
            self.res_cdda['len_adapt'].append(drift_idx-tr_start_idx)

            # reset values
            self._reset_parameters()
            self.warn_prd     = []
            self.res_pred_tmp = []
        # end if

        return tr_start_idx

    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_min    = np.inf
        self.s_min    = np.inf

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
            print(f'tr_start_idx: {} / tr_end_idx: {}')
            
            # add values into dict containing results of cdda
            self.res_cdda['time_idx'].extend(X_te.index)
            self.res_cdda['y_real_list'].extend(y_te)
            self.res_cdda['y_pred_list'].extend(y_pred_te)
            self.res_cdda['res_pred_list'].extend(res_pred_idx)

            # add prediction results for cdd
            self.res_pred_tmp.extend(res_pred_idx)

            if len(self.res_pred_tmp) >= self.clt_idx:
                #print('length of predictions:', len(self.res_pred_tmp))

                # compute err. rate and std. dev.
                p_idx = 1 - sum(self.res_pred_tmp)/len(self.res_pred_tmp)
                s_idx = np.sqrt(p_idx*(1-p_idx)/len((self.res_pred_tmp)))

                # update p_min and s_min if p_idx + s_idx is lower than p_min + s_min
                self.p_min = p_idx if p_idx + s_idx < self.p_min + self.s_min else self.p_min
                self.s_min = s_idx if p_idx + s_idx < self.p_min + self.s_min else self.s_min

                # evaluate state of the concept
                self.state = self._detect_drift(p_idx   = p_idx,
                                                s_idx   = s_idx, 
                                                p_min   = self.p_min, 
                                                s_min   = self.s_min, 
                                                alpha_w = self.alpha_w, 
                                                alpha_d = self.alpha_d)

                # TODO: tr_start_idx, tr_end_idx 위치 확인 / online, offline learning 모두 포괄가능하게
                # TODO: tr_start_idx, tr_end_idx 모두 함수의 return값으로?!
                # set the start index of updated training set
                tr_start_idx = self._adapt_drift(state        = self.state, 
                                                 min_len_tr   = min_len_tr,
                                                 tr_start_idx = tr_start_idx, 
                                                 tr_end_idx   = tr_end_idx, 
                                                 te_end_idx   = te_end_idx)
            # end if
            
            # set the end index of updated training set
            tr_end_idx += len_batch
        # end while

        return None