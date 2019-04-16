#%%
import numpy as np
import Utility as util
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3 as sql
import ast
from scipy.linalg import svd
from scipy.sparse.linalg import svds

# %% Helper-Functions
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

def get_P(user_data,k):
    U,sigma,_ = svds(user_data, k=k)
    P = np.dot(U,np.diag(sigma))
    return np.asarray(P)

def get_Q(user_data,k):
    _, _, Q = svds(user_data,k=k)
    Q=Q.T
    return np.asarray(Q)

def MF():

def update(p,Q,u,i,r,k,rw,rc,alpha,lam):
    omega = np.sum(rw[:,i]==r)
    rui = rc[u][i]

    P_ = np.reshape(p,(-1,1)).T
    Q_ = np.reshape(Q[i,:],(-1,1))
    rcap = np.dot(P_,Q_)
    gact = g_elo_actual(rui,r)
    gexp = g_logistic(rcap,r)
    grad = omega*Q[i,:]*(gact-gexp) + lam*(P_)
    P_ -= alpha*grad
    return np.reshape(P_.T,(k))

def update2(P,Q,u,i,r,f,rw,rc,alpha,lam):
    omega = np.sum(rw[:,i]==r)
    rui = rc[u][i]
    P_ = np.reshape(P[u,:],(-1,1)).T
    Q_ = np.reshape(Q[i,:],(-1,1))
    rcap = np.dot(P_,Q_)
    gact = g_elo_actual(rui,r)
    gexp = g_logistic(rcap,r)
    grad = omega*Q[i,f]*(gact-gexp)
    return alpha*grad


def rapare(P,Q,rc,rw,rmax,epochs,k,alpha,lam):
    res_mse = []
    for epc in range(epochs):
        for u in range(rc.shape[0]):
            for i in range(rc.shape[1]):
                if(rc[u][i]!= 0):
                    for r in range (1,rmax+1):
                        P[u] = update(P[u],Q,u,i,r,k,rw,rc,alpha,lam)
                        # for f in range(0,k):
                        #     P[u][f] -= update2(P,Q,u,i,r,f,rw,rc,alpha,lam)
        print(epc)
        r_est = np.dot(P,Q.T)
        r_est[r_est<=0]=0
        r_est[r_est>=5]=5
        r_est[rc==0]=0
        # r_est = np.round(r_est)
        mse = np.sqrt(np.sum((r_est-rc)**2)/np.sum(rc>0))
        print('RMSE:' , mse )
        res_mse.append(mse)
    save_plot(res_mse,epochs,'training2.png')
    return P,res_mse

def save_plot(mse,epochs,f_name):
    path='./'+f_name
    x = np.arange(0,epochs,1)
    fig = plt.figure()
    plt.plot(x,mse)
    plt.plot()
    fig.savefig(path,dpi=fig.dpi)

#%% Hyperparams
epochs = 100
lam = 0
k = 20 # number of components
alpha = 0.1
rmax = 5


#%% get_data
data,data_movies = util.get_user_movie_rating()
r  = np.asarray(data_movies,dtype=np.float32)[:,1:]
rc = np.asarray(data_movies[:1000],dtype=np.float32)[:,1:]
rw = np.asarray(data_movies[1000:],dtype=np.float32)[:,1:]

#%% computing Latent factors
# P = np.random.rand(rc.shape[0],k)
# P = np.load('500epc_p.npy')

P = get_P(rc,k)
Q = get_Q(r,k)
#%%
P,Rmse = rapare(P,Q,rc,rw,rmax,epochs,k,alpha,lam)
# np.save('600epc_p.npy',P)
# r_est = np.dot(P,Q.T)
# r_est[r_est<=0]=0
# r_est[r_est>=5]=5
# r_est = np.round(r_est)
# r_est[rc==0]=0
# np.sum(r_est==rc)
# np.sum(rc>-1)
# mse = np.sqrt(np.sum((r_est-rc)**2)/np.sum(rc>0))
# print('RMSE:' , mse )
# REVIEW:
# x,y = np.asarray(np.where(rc>0))
# x.shape
# y.shape
