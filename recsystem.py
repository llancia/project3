'''
Lorenzo Lancia
Recommender systema using SVD Decomposition

Project 3 of ADM Course
Master Degree in Data Science
University of Rome, La Sapienza
lorentz90[AT]gmail[DOT]com
'''
from random import shuffle
from scipy import sparse, linalg

import numpy as np

from numba import jit


def load_data(filename, separator="\t"):
    '''
    Funtion that load a file name e returns list of list [user,movie,rating]
    '''
    f = open(filename,"r")
    items = []
    for line in f.readlines():
        line = line.split(separator)
        u= int(line[0])
        m= int(line[1])
        r= int(line[2])
        items.append([u,m,r])
    return items


def split_test_train(data, rate):
    '''
    Split data in random trainig set and test set, returns two list of list
    '''
    shuffle(data)
    n_of_training = int(len(data) * rate)
    return data[:n_of_training], data[-n_of_training:]

    
    
def create_matrix(train_data):
    '''
    Build User x Item = Rating Matrix for svd training
    '''
    row, col, data = zip(*train_data)
    mtx = sparse.coo_matrix((data, (np.array(row)-1, np.array(col)-1)))
    return mtx.tocsr()

def find_user_and_item_avg(M):
    '''
    Given a scipy.sparse csr matrix return mean, and row and col deviation
    '''
    useravg = [float(M[i].sum()) /M[i].nnz for i in  range(M.shape[0])]
    itemavg = [float(M[:,i].sum()) /M[:,i].nnz if M[:, i].sum() !=0 else 0 for i in range(M.shape[1])]

    return useravg, itemavg
  

def norm_matrix(R, useravg, itemavg):
   
    Matrix = R.toarray().astype(np.float32, copy=False)
    indices_zero = np.nonzero(Matrix == 0)
    for i in range(len(indices_zero[0])):
        Matrix[indices_zero[0][i]][indices_zero[1][i]] = itemavg[indices_zero[1][i]]
    for u_id in range(R.shape[0]):
        Matrix[u_id] = Matrix[u_id]-useravg[u_id]
    return Matrix

def train_complete_SVD(M, K=9):
    useravg, itemavg = find_user_and_item_avg(M)
    R_norm = norm_matrix(M, useravg, itemavg)
    U, s, V = linalg.svd( R_norm, full_matrices=False )
    m_user,n_movies = R_norm.shape
    new_s = s[:K]
    sigma_1_2 = linalg.diagsvd(np.sqrt(new_s), K, K)
    U_tilde = np.dot(U[:,:K], sigma_1_2)
    V_tilde = np.dot(sigma_1_2, V[:K,:])
    return U_tilde, V_tilde

def special_svd(M, K=9):
    useravg, itemavg = find_user_and_item_avg(M)
    R_norm = norm_matrix(M, useravg, itemavg)
    U, s, V = linalg.svd( R_norm, full_matrices=False )
    m_user,n_movies = R_norm.shape
    new_s = s[:K]
    sigma = linalg.diagsvd(new_s, K, K)
    return U[:,:K], V[:K,:], sigma_1_2


@jit(nopython=True)
def funk_predict(user, item, U, V, useravg ):
    
    rating =useravg[user]
    #Dot product
    for f in range(U.shape[0]):
        rating += U[f, user] * V[f, item]
        
    if rating > 5: rating = 5
    elif rating < 1: rating = 1
    return rating


def validate_with_testdata(U_tilde, V_tilde, testdata, useravg):
    num=0
    mse=0
    for item in testdata:
        user = item[0]-1
        movie = item[1]-1
        r = item[2]
        estimated = U_tilde[user].dot(V_tilde[:,movie])+useravg[user]
        mse += (r-estimated)**2
        num+=1
    return np.sqrt(mse/num)




####### CONVERGENCE FUNK SVD #####

def init_U_V (n_users, n_items, n_features, INIT_VALUE ):
    U = np.empty((n_features, n_users))
    V = np.empty((n_features, n_items))
    U[:] = INIT_VALUE
    V[:] = INIT_VALUE
    return U,V


@jit(nopython=True)
def gradient_descent(feature, A_row, A_col, A_data, Matrix_U, Matrix_V, K,useravg ):
    REG_TERM = 0.015
    LAMBDA = 0.001
    squared_error = 0
    for k in range(len(A_data)):
        u_id = A_row[k]
        i_id = A_col[k]
        rating = A_data[k]
        err = rating - funk_predict(u_id, i_id, Matrix_U, Matrix_V, useravg)
        squared_error += err ** 2

        U_af = Matrix_U[feature, u_id]
        V_if = Matrix_V[feature, i_id]
        
        Matrix_U[feature, u_id] += LAMBDA * (err * V_if - REG_TERM * U_af)
        Matrix_V[feature, i_id] += LAMBDA * (err * U_af - REG_TERM * V_if)

    return squared_error





def my_train_SVD(M, NUM_FEATURES=20, EPSILON = 0.00005, MIN_ITERATIONS = 200, INIT_VALUE = 0.1):
    """
    
    """
    useravg, itemavg = find_user_and_item_avg(M)
    Matrix_U, Matrix_V  = init_U_V(M.shape[0], M.shape[1], NUM_FEATURES, INIT_VALUE )

    R = M.tocoo()
    rmse = 0
    last_rmse = 0
   
    num_ratings = len(R.data)
    for feature in xrange(NUM_FEATURES):
        iter = 0
        converged = False
        while not converged:
            squared_error = gradient_descent(feature, R.row, R.col, R.data, Matrix_U, Matrix_V, NUM_FEATURES, useravg )
            rmse = (squared_error / num_ratings) ** 0.5
            if  (iter > MIN_ITERATIONS) or  (last_rmse - rmse < EPSILON):
                converged = True
            last_rmse = rmse
            iter += 1
    print "TRAINED!"
    return Matrix_U, Matrix_V

def funk_validate_with_testdata(U_tilde, V_tilde, testdata, useravg):
    num=0
    mse=0
    for item in testdata:
        user = item[0]-1
        movie = item[1]-1
        r = item[2]
        estimated =funk_predict(user,movie, U_tilde, V_tilde, useravg)
        mse += (r-estimated)**2
        num+=1
    return np.sqrt(mse/num)


def read_new_user(filename):
    f = open(filename,"r")
    items = []
    for line in f.readlines():
        line = line.split(separator)
        m= int(line[0])
        r= int(line[1])
        items.append([m,r])
    return items

def reccomend_for_a_new_user(SIGMA, Vh, N_U):
    


