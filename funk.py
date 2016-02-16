# Requires Movielens 100k data 
import numpy as np
from numba import jit
from scipy.sparse import csr_matrix, dok_matrix

'''
Creating a sparse matrix, User vs Item = rating
'''
def create_matrix(filename):
    f = open(filename,"r")
    links=[]
    dim = [0,0]
    for line in f.readlines():
        line = line.split("\t")
        u = float(line[0])-1
        v = float(line[1])-1
        r = float(line[2])
        links.append((u,v,r))
        if u > dim[0]:
            dim[0] = u
        if v > dim[1]:
            dim[1] = v
    B = dok_matrix((dim[0]+1,dim[1]+1), dtype=float)
    B.update({(link[0],link[1]) : link[2] for link in links})
    return B.tocsr()


def init_U_V (n_users, n_items, n_features, INIT_VALUE ):
    U = np.empty((n_features, n_users))
    V = np.empty((n_features, n_items))
    U[:] = INIT_VALUE
    V[:] = INIT_VALUE
    return U,V

def find_useritem_bias(M, mu):
    userbias =[(M[i].tocoo().data-mu).mean() for i in range(M.shape[0])]
    itembias =[(M.transpose()[i].tocoo().data-mu).mean() for i in range(M.shape[1])]
    itembias =np.array([0. if np.isnan(itemb) else itemb for itemb in itembias])
    return userbias, itembias

@jit(nopython=True)
def predict(u_i, i_a, U, V, b_u, b_i, mu ):
    
    rating = b_u[u_i] 
    #Dot product
    for f in range(U.shape[0]):
        rating += U[f, u_i] * V[f, i_a]
        
    if rating > 5: rating = 5
    elif rating < 1: rating = 1
    return rating

    
@jit(nopython=True)
def gradient_descent(feature, A_row, A_col, A_data, Matrix_U, Matrix_V, K,userbias, itembias, mu ):
    REG_TERM = 0.015
    LAMBDA = 0.001
    squared_error = 0
    for k in range(len(A_data)):
        u_id = A_row[k]
        i_id = A_col[k]
        rating = A_data[k]
        err = rating - predict(u_id, i_id, Matrix_U, Matrix_V, userbias, itembias, mu)
        squared_error += err ** 2

        U_af = Matrix_U[feature, u_id]
        V_if = Matrix_V[feature, i_id]
        
        Matrix_U[feature, u_id] += LAMBDA * (err * V_if - REG_TERM * U_af)
        Matrix_V[feature, i_id] += LAMBDA * (err * U_af - REG_TERM * V_if)

    return squared_error





def my_train_SVD(A_row, A_col, A_data, Matrix_U, Matrix_V, NUM_FEATURES, userbias, itembias, mu ):
    """
    
    """
    EPSILON = 0.00005
    MIN_ITERATIONS = 200
    rmse = 0
    last_rmse = 0
   
    num_ratings = len(A_data)
    for feature in xrange(NUM_FEATURES):
        iter = 0
        converged = False
        while not converged:
            squared_error = gradient_descent(feature, A_row, A_col, A_data, Matrix_U, Matrix_V, NUM_FEATURES, userbias, itembias, mu )
            rmse = (squared_error / num_ratings) ** 0.5
            if  (iter > MIN_ITERATIONS) or  (last_rmse - rmse < EPSILON):
                converged = True
            last_rmse = rmse
            iter += 1
    print "TRAINED!"
    return 


def validate_with_testdata(filename, U, V, userbias, itembias, mu):
    f = open(filename,"r")
    mse=0
    R_0 = 0
    for line in f.readlines():
        line = line.split("\t")
        c = int(line[0])-1
        p = int(line[1])-1
        r = int(line[2])
        mse += (r- predict(c, p, U, V, userbias, itembias , mu))**2
        R_0+=1
        
       
    return np.sqrt(mse/R_0)

LAMBDA = 0.02
INIT_VALUE = 0.1
K = 20



A = create_matrix("./ml-100k/u1.base")
useravg = A.sum(1) / (A != 0).sum(1)
useravg =[float(avg[0][0]) for avg in useravg]
U,V = init_U_V(A.shape[0], A.shape[1], K, INIT_VALUE )
R = A.tocoo()
mu = R.data.mean()
userbias, itembias = find_useritem_bias(A,mu)

my_train_SVD(R.row, R.col, R.data, U, V, K, useravg, itembias, mu )




print validate_with_testdata("./ml-100k/u1.test", U, V, useravg,itembias, mu)
