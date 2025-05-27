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

def run_ml_model_pipeline(X_tr, y_tr, X_te, y_te, scaler, ml_mdl):
    # if online learning is available
    if isinstance(scaler, OnlineStandardScaler) and hasattr(ml_mdl, 'partial_fit'):
        # run online learning
        pass

    else:
        # offline scaling
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        
        # train the model
        ml_mdl.fit(X_tr, y_tr)
    # end if


    # predict testset
    y_pred_te = ml_mdl.predict(X_te)

    return ml_mdl, y_pred_te