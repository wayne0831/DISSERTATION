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
from src.util.preprocess import OnlineStandardScaler
from src.util.metric import hit_ratio
from src.util.visualization import *
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model   import LinearRegression, Lasso, LogisticRegression, Ridge, SGDRegressor
from sklearn.metrics import accuracy_score, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
import pandas as pd
import numpy as np
import os
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

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

tr_idx_list = list(range(init_tr_start_idx, init_tr_end_idx))
te_idx_list = list(range(init_tr_end_idx, len(X)))

# set training/test set
X_tr, y_tr = X.iloc[tr_idx_list], y[tr_idx_list]
X_te, y_te = X.iloc[te_idx_list], y[te_idx_list]

print(X_tr.shape, X_te.shape)

##################################################################################################################################################################
# run model
##################################################################################################################################################################

scl      = StandardScaler()
X_tr_scl = scl.fit_transform(X_tr)
X_te_scl = scl.fit_transform(X_te)

mdl = SGDRegressor(random_state=42, max_iter=1000, tol=1e-3)
mdl.fit(X_tr_scl, y_tr)

y_pred = mdl.predict(X_te_scl)

df_pred = pd.DataFrame(y_pred)
df_pred.to_csv('./result/prediction/cdda/static_pred.csv')

