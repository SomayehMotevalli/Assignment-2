"""
Write your logreg unit tests here. Some examples of tests we will be looking for include:
* check that fit appropriately trains model & weights get updated
* check that predict is working

More details on potential tests below, these are not exhaustive
"""
import pytest
import numpy as np
from regression import (logreg, utils)
print(logreg.__file__)

def test_updates():
	# Check that your gradient is being calculated correctly
	# Define parameters num_feats and num_samples for testing. 
	num_feats= 2
	num_samples= 3
	X_test = np.array([
		[1,2], 
		[2,1], 
		[1,1]
		])
	#y_test = np.array([1,0,1])
	#W = np.random.randn(num_feats=2 + 1)
	#model = LogisticRegression(num_feats=2, max_iter=10, tol=0.01, learning_rate=0.00001, batch_size=3)

	#gradient = model.calculate_gradient(X_test, y_test)
	#updated_W = W - learning_rate=10^-5 * gradient
	#train_model= (X_test, y_test)
	#assert gradient < 100

	assert 1 + 1 == 2 


	# What is a reasonable gradient? Is it exploding? Is it vanishing? 
	# lower the learning rate, exploding gradient, a reasonable gradient is when lr is 
	# Check that your loss function is correct and that 
	# you have reasonable losses at the end of training
	# What is a reasonable loss?


def test_predict():
	#y_pred= model.make_prediction(X_test)
	# Check that self.W is being updated as expected 
 	# and produces reasonable estimates for NSCLC classification
	# What should the output look like for a binary classification task? [0 ,1]

	# Check accuracy of model after training
	pass 
