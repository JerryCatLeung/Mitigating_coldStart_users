#%%
import numpy as np
import Utility as util
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3 as sql
import ast
from scipy.linalg import svd

# %%
data,data_movies = get_user_movie_rating()
cold_users = data_movies[:1000]
warm_users = data_movies[1000:]
cold_users[1]
R = cold_users
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds
U,sigma,_ = svds(R_demeaned, k = 50)
P = np.dot(U,np.diag(sigma))


R = warm_users
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
_, _, Q = svds(R_demeaned, k = 50)

Q=Q.T

lam,alpha = 1,0.1 #change it please
def g_elo_actual(a,b):
    if(a>b):
        return 1
    if(a==b):
        return 0.5
    else:
        return 0
def g_logistic(a,b):
    return 1/(1+np.exp(-(a-b)))
def g_linear(a,b):
    return (a-b)

def update(p,i,j,r,f):
    count_r =  np.count_nonzero(warm_users[:][j]==r)
    rui = cold_users[i][j]
    P_ = np.reshape(P[i][:],(-1,1)).T
    Q_ = np.reshape(Q[:][j],(-1,1))
    # print (P_.shape,Q_.shape)
    rcap = np.dot(P_,Q_)
    grad = 2 * Q[i][f] * count_r * (g_linear(rui,r) - g_linear(rcap,r)) + (2 * lam * P[i][f])
    return (p - grad)

def func(cold_users,P,Q,r_max = 5,k=50):
    for temp in range(2):
        for i in range(len(cold_users)):
            for j in range(len(cold_users[0])):
                if (cold_users[i][j] != 0):
                    for r in range(r_max):
                        for f in range(k):
                            P[i][f] = update(P[i][f],i,j,r,f)
    return P

P = func(cold_users,P,Q)
