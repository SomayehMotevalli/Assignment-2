"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression import (logreg, utils)
from sklearn.preprocessing import StandardScaler

def test_updates():

    # load data with default settings
    X_train, X_val, y_train, y_val = utils.loadDataset(features=['Penicillin V Potassium 500 MG', 'Computed tomography of chest and abdomen', 
                                                                  'Plain chest X-ray (procedure)',  'Low Density Lipoprotein Cholesterol',
                                                                  'Creatinine', 'AGE_DIAGNOSIS'], split_percent=0.8, split_state=42)

    # scale data since values vary across features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.transform (X_val)
    
    print(X_train.shape, X_val.shape, y_val.shape, y_train.shape)


    """
    # for testing purposes once you've added your code
    # CAUTION & HINT: hyperparameters have not been optimized

    log_model = logreg.LogisticRegression(num_feats=2, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    log_model.plot_loss_history()
            
    """
    # Check that your gradient is being calculated correctly

    log_model = logreg.LogisticRegression(num_feats=6, max_iter=10, tol=0.01, learning_rate=0.01, batch_size=12)
    log_model.train_model(X_train, y_train, X_val, y_val)
    for grad in log_model.gradient_history:
     assert np.linalg.norm(grad) <10
     assert np.linalg.norm(grad) >0.001
     assert 0.001 <np.linalg.norm(grad)< 10
     assert log_model.loss_history_val[0] > log_model.loss_history_val[-1]
     # assert log_model.loss_history_val [-1] < 0.5 # the final element of my validation loss is not less than 0.5
     

# What is a reasonable gradient? it is between 0.001 and 10.
    0.001 <np.linalg.norm(grad)< 10
#  Is it exploding? It is not exploding (> 0.001).
# Is it vanishing? It is not vanishing (<10).
# Check that your loss function is correct and that 
# you have reasonable losses at the end of training
# What is a reasonable loss? 0.6676


def test_predict():
	#y_pred= model.make_prediction(X_test)
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output look like for a binary classification task? [0 ,1]

	# Check accuracy of model after training
	pass 
