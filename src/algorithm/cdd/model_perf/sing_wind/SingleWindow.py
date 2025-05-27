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
        **kwargs: {alpha_w: float, alpha_d: float, warm_start: int}

        :param alpha_w:     warning conf.level                    (rng: >= 1)   (type: float)            
        :param alpha_d:     drift conf.level                      (rng: >= 1)   (type: float)
        :param warm_start:  min. num. of data points to obey CLT  (rng: >= 30)  (type: int)
        """
        # values to be reset after drift detection
        self.p_min = np.inf   # min. err. rate of the ML model
        self.s_min = np.inf   # min. std. dev. of the ML model

        # hyperparameters of DDM
        self.alpha_w    = kwargs['alpha_w']
        self.alpha_d    = kwargs['alpha_d'] 
        self.warm_start = kwargs['warm_start'] 

        # state / warning period / prediction results for each cdd
        self.state        = 0   # 0: stable / 1: warning / 2: drift
        self.warn_prd     = []
        self.res_pred_tmp = []

    def _reset_parameters(self):
        """
        Reset parameters for next iteration of concept drift detection
        """
        self.p_min    = np.inf
        self.s_min    = np.inf

        return None

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
    
    def run_cdd(self, res_prd, idx):
        # add prediction results for cdd
        self.res_pred_tmp.extend(res_prd)

        if len(self.res_pred_tmp) >= self.warm_start:
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
            if self.state == 0:
                pass
            elif self.state == 1:
                self.warn_prd.append(idx)
            elif self.state == 2:
                self._reset_parameters()
            # end if
        # end if
        
        return None
