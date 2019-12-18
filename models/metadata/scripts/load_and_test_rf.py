import numpy as np
import pandas as pd
import pickle

from sklearn.metrics import mean_squared_error

filename = 'rf_score_first.pkl'

rf_loaded = pickle.load(open(filename, 'rb'))

random_index = np.random.randint(0, len(X_test) - 1)

# =============================================================================
# Data preprocessing and loading test data
test_vector = X_test.iloc[random_index, :]
test_vector = test_vector.values
test_vector = test_vector.reshape(1, -1)
test_label = y_test.iloc[random_index]


######## i have to write all the lines to preprocess the input data
# =============================================================================

prediction = rf_loaded.predict(test_vector)
error = mean_squared_error(test_label, prediction)