import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold, RandomizedSearchCV #, train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import pickle

########## XGB MACHINE ################

#multioutputregressor = MultiOutputRegressor(xgb.XGBRegressor(objective='reg:squarederror')).fit(x_train, y_train)

xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train, y_train)

import scipy.stats as st

one_to_left = st.beta(10, 1)  
from_zero_positive = st.expon(0, 50)
params = {  
    "n_estimators": st.randint(3, 40),
    "max_depth": st.randint(3, 40),
    "learning_rate": st.uniform(0.05, 0.4),
    "colsample_bytree": one_to_left,
    "subsample": one_to_left,
    "gamma": st.uniform(0, 10),
    'reg_alpha': from_zero_positive,
    "min_child_weight": from_zero_positive,
}

search = RandomizedSearchCV(xg_reg, param_distributions=params, 
                            random_state=42, n_iter=200, cv=5, verbose=1, 
                            n_jobs=-1, return_train_score=True, 
                            scoring = 'neg_mean_squared_error')

search.fit(X_train, y_train)

print('\nThe best parameters for the model are: \n{} '.format(search.best_params_))
print('\nThe best score for the model is: {} '.format(search.best_score_))


#error = np.mean((multioutputregressor.predict(x_test) - y_test)**2, axis=0)
preds = xg_reg.predict(X_test)
preds_RandomSearch = search.best_estimator_.predict(X_test)
error = np.sqrt(mean_squared_error(y_test, preds_RandomSearch))

print('The MSE for the model is {}'.format(error))
print(explained_variance_score(y_test, preds_RandomSearch))

search._best_estimator_.save_model('XGB_best_estimator.model')
