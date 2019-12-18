import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

import pickle

from pprint import pprint

from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
# Loading the new data
X_train = pd.read_csv('X_train_final')
X_test = pd.read_csv('X_test_final')
y_train = pd.read_csv('y_train_final', header=None)
y_test = pd.read_csv('y_test_final', header=None)

# =============================================================================
# Random Forest
# =============================================================================
n_estimators = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [5, 7, 8, 9, 10, 11, 12, 13, 14, 15, None]
# Minimum number of samples required to split a node
min_samples_split = [2, 3, 4, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 3, 4, 5]
# Method of selecting samples for training each tree
bootstrap = [True]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)

rf = RandomForestRegressor()
search = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, 
                             n_iter = 300, cv = 5, verbose=10, random_state=42, 
                             n_jobs = -1, scoring='neg_mean_squared_error')
search.fit(X_train, y_train)
pprint(search.best_params_)
best_random = search.best_estimator_
print('The best score is {}'.format(search.best_score_))

print('R2_score for the model: ')
print(r2_score(y_test, best_random.predict(X_test)))
print('mean squared error for the model: ')
print(mean_squared_error(y_test, best_random.predict(X_test)))
# =============================================================================

# Saving the model
# =============================================================================
with open('rf_score_' + str(r2_score(y_test, best_random.predict(X_test))) +'.pkl', 'wb') as fid:
    pickle.dump(best_random, fid)  
# =============================================================================










