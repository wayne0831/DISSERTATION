##################################################################################################################################################################
# PROJECT: DISSERTATION
# CHAPTER: Util
# SECTION: Common
# AUTHOR: Yang et al.
# DATE: since 25.06.03
##################################################################################################################################################################

##################################################################################################################################################################
# import libraries
##################################################################################################################################################################

from sklearn.preprocessing import StandardScaler
from src.util.preprocess import OnlineStandardScaler
import pandas as pd
import numpy as np
from scipy import stats
from itertools import product
import time

##################################################################################################################################################################
# user-defined utility functions
##################################################################################################################################################################

# def run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl):
#     # scale dataset
#     if scaler is not None:
#         X_tr = scaler.fit_transform(X_tr)
#         X_te = scaler.transform(X_te)
#     # end if

#     # train the model
#     ml_mdl.fit(X_tr, y_tr)

#     # predict testset 
#     y_pred_te = ml_mdl.predict(X_te)

#     return ml_mdl, y_pred_te

# TODO: 함수 활용 여부 고민 필요
def set_ml_dataset(tr_start_idx, tr_end_idx, len_batch, X, y, X_cum, y_cum):
    """
    Set dataset for ml model
    """

    # set test set index
    te_start_idx = tr_end_idx
    te_end_idx   = min(tr_end_idx + len_batch, len(X))

    tr_idx_list = list(range(tr_start_idx, tr_end_idx))
    te_idx_list = list(range(te_start_idx, te_end_idx)) # TODO: te_idx_list, te_end_idx 비교 잘하자

    # set training/test set
    X_tr, y_tr = X[tr_idx_list], y[tr_idx_list]
    X_te, y_te = X[te_idx_list], y[te_idx_list]

    # cumulate incoming data points
    X_cum = np.concatenate([X_cum, X_tr]) if X_cum is not None else X_tr
    y_cum = np.concatenate([y_cum, y_tr]) if y_cum is not None else y_tr

    return X_cum, y_cum, X_tr, y_tr, X_te, y_te

# TODO: set_dataset 제외 코드 재작성
def run_ml_model(X_cum, y_cum, X_tr, y_tr, X_te, y_te, scaler, ml_mdl, prob_type, perf_bnd):
    """
    Run ml model pipeline: scale dataset, train ml model, predict data
    """
    # if online learning is available
    if isinstance(scaler, OnlineStandardScaler) and hasattr(ml_mdl, 'partial_fit'):
        # partially scale dataset
        scaler.partial_fit(X_tr)
        X_tr_scl = scaler.fit_transform(X_tr)
        X_te_scl = scaler.transform(X_te)

        # partially fit the ml model
        ml_mdl.partial_fit(X_tr_scl, y_tr, classes=np.unique(y))

        # predic the test sets
        y_pred_te = ml_mdl.predict(X_te_scl)
    else:
        # offline scaling
        X_tr_scl = scaler.fit_transform(X_cum)
        X_te_scl = scaler.transform(X_te)
        
        # train the model
        ml_mdl.fit(X_tr_scl, y_cum)
    # end if

    # predict testset
    y_pred_te = ml_mdl.predict(X_te_scl)

    # extract prediction results
    if prob_type == 'CLF':
        res_pred_idx = [1 if pred == real else 0 for pred, real in zip(y_pred_te, y_te)] 
    elif prob_type == 'REG':
        res_pred_idx = [1 if abs(pred - real) <= perf_bnd else 0 for pred, real in zip(y_pred_te, y_te)] 
    # end if

    return y_pred_te, res_pred_idx