import numpy as np
from utils import *

def preprocess(X, Y):
	''' 
	X = input feature matrix [N X D] 
	Y = output values [N X 1]
	Convert data X, Y obtained from read_data() to a usable format by gradient descent function
	Return the processed X, Y that can be directly passed to grad_descent function
	NOTE: X has first column denote index of data point. Ignore that column 
	and add constant 1 instead (for bias part of feature set)
	'''
	dimension= X.shape
	N = dimension[0]
	M = dimension[1]
	X_modi = np.ones((N,1))
	
	
	for column in range(1,M):
		if type(X[1,column]) != str:
			mean = np.mean(X[:, column])
			std = np.std(X[:, column])
			temp = np.zeros((N,1))
			for row in range(N):
				temp[row,0] = (X[row, column] - mean)/std

			X_modi = np.concatenate((X_modi, temp), axis = 1)

			

		else:
			all_values = X[:,column]
			all_labels = list(set(all_values))
			newX = one_hot_encode(all_values, all_labels)
			X_modi = np.concatenate((X_modi, newX), axis = 1)
			

			

	
	return X_modi.astype(np.float32), Y.astype(np.float32)

	

def grad_ridge(W, X, Y, _lambda):
	'''  
	W = weight vector [D X 1]
	X = input feature matrix [N X D]
	Y = output values [N X 1]
	_lambda = scalar parameter lambda
	Return the gradient of ridge objective function (||Y - X W||^2  + lambda*||w||^2 )
	'''
	
	X = X.astype(np.float32)
	Y = Y.astype(np.float32)
	W = W.astype(np.float32)
	
	D = W.shape[0]
	N = X.shape[0]

	

	grad_desc = (-2) * np.matmul(np.transpose(X), np.subtract(Y, np.matmul(X, W))) + 2 * _lambda * W

	return grad_desc

	

def ridge_grad_descent(X, Y, _lambda, max_iter=30000, lr=0.00001, epsilon = 1e-4):
	''' 
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	lr 			= learning rate
	epsilon 	= gradient norm below which we can say that the algorithm has converged 
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 
	NOTE: You may precompure some values to make computation faster
	'''
	N = X.shape[0]
	D = X.shape[1]


	W = np.ones((D,1))
	


	def isconverged(W_prev, W_new, epsilon):
		for i in range(len(W_prev)):
			if abs(W_prev[i] - W_new[i]) > epsilon:
				return False

		return True

	for i in range(max_iter):
		
		W_old = W.copy()
		W -= lr * grad_ridge(W, X, Y, _lambda)
		if isconverged(W_old, W, epsilon) == True:
			return W

	



def k_fold_cross_validation(X, Y, k, lambdas, algo):
	''' 
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	k 			= number of splits to perform while doing kfold cross validation
	lambdas 	= list of scalar parameter lambda
	algo 		= one of {coord_grad_descent, ridge_grad_descent}
	Return a list of average SSE values (on validation set) across various datasets obtained from k equal splits in X, Y 
	on each of the lambdas given 
	'''
	N = X.shape[0]
	D = X.shape[1]

	avg_sse = []
	split_size = N//k
	no_remaining_data = N%k
	
	#creating a copy to avoid risks
	X_temp = X.copy()
	Y_temp = Y.copy()
	all_tests_X = []
	all_train_X = []
	all_tests_Y = []
	all_train_Y = []

	

	all_blocks_X = np.split(X_temp[:split_size*k, :], k, axis = 0)
	all_blocks_Y = np.split(Y_temp[:split_size*k, :], k, axis = 0)

	
	
	for i in range(k):
		all_blocks_X_temp = all_blocks_X.copy()
		all_blocks_Y_temp = all_blocks_Y.copy()

		test_set_x = all_blocks_X_temp[i]
		
		train_set_x = np.delete(all_blocks_X_temp, [i], axis = 0)
		merged_train_set_x = np.concatenate(tuple(train_set_x), axis = 0)

		
		


		all_tests_X.append(test_set_x)
		all_train_X.append(merged_train_set_x)

		test_set_y = all_blocks_Y_temp[i]
		train_set_y = np.delete(all_blocks_Y_temp, [i], axis = 0)
		merged_train_set_y = np.concatenate(tuple(train_set_y), axis = 0)

		all_tests_Y.append(test_set_y)
		all_train_Y.append(merged_train_set_y)
		

	for l in range(len(lambdas)):
		all_sse = []

		for i in range(k):
			
			W = algo(all_train_X[i], all_train_Y[i], lambdas[l])
			all_sse.append(sse(all_tests_X[i], all_tests_Y[i], W))

		avg_sse.append(sum(all_sse)/len(all_sse))

	return avg_sse










	

def coord_grad_descent(X, Y, _lambda, max_iter=1000):
	''' 
	X 			= input feature matrix [N X D]
	Y 			= output values [N X 1]
	_lambda 	= scalar parameter lambda
	max_iter 	= maximum number of iterations of gradient descent to run in case of no convergence
	Return the trained weight vector [D X 1] after performing gradient descent using Ridge Loss Function 

	'''
	N = X.shape[0]
	D = X.shape[1]

	W = np.ones((D, 1))

	for i in range(max_iter):
		

		
		X_temp = X.copy()
		Y_temp = Y.copy()
		a = 0
		for j in range(D):
				
				a += np.sum(X_temp[:, j]  * W[j,0])

		r = np.sum(Y[:,0])
		new_W = np.ones((D,1))

		for d in range(D):
			
			
					
			p = a - np.sum(X_temp[:,d] * W[d])
			q = np.sum(X_temp[:,d])
			if q == 0:
				W[d,0] = 0
			if W[d,0] > 0:
				new_W[d,0] = ((_lambda/2) - q * (p + r)) /q * q
			elif W[d,0] < 0:
				new_W[d,0] = (-(_lambda/2) - q* (p + r) ) / q * q
			

		

	return W



	 





	

if __name__ == "__main__":
	# Do your testing for Kfold Cross Validation in by experimenting with the code below 
	X, Y = read_data("./dataset/train.csv")
	X, Y = preprocess(X, Y)
	trainX, trainY, testX, testY = separate_data(X, Y)
	
	lambdas = [...] # Assign a suitable list Task 5 need best SSE on test data so tune lambda accordingly
	scores = k_fold_cross_validation(trainX, trainY, 6, lambdas, ridge_grad_descent)
	plot_kfold(lambdas, scores)
