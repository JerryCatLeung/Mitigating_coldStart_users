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
    #print(sigma)
    P = np.dot(U,np.diag(sigma))
    return np.asarray(P)

def get_Q(user_data,k):
    _, _, Q = svds(user_data,k=k)
    Q=Q.T
    return np.asarray(Q)

def update(p,Q,u,i,r,k,rw,rc,alpha,lam):
    omega = np.sum(rw[:,i]==r)
    rui = rc[u][i]
    P_ = np.reshape(p,(-1,1)).T
    Q_ = np.reshape(Q[i,:],(-1,1))
    rcap = np.dot(P_,Q_)
    gact = g_elo_actual(rui,r)
    gexp = g_logistic(rcap,r)
    grad = omega * Q[i,:]*(gact-gexp) + lam * (P_)
    P_  += alpha * grad
    return np.reshape(P_.T,(k))


def rapare(P,Q,rc,rw,rmax,epochs,k,alpha,lam,res_mse,lam_step,lam_change,alpha_step,alpha_change):
    old_mse = 0
    for epc in range(epochs):
        if(epc%lam_step==0):
            lam+=lam_change
        if(epc%alpha_step==0):
            alpha -= alpha_change*alpha
        for u in range(rc.shape[0]):
            for i in range(rc.shape[1]):
                if(rc[u][i]!= 0):
                    for r in range (1,rmax+1):
                        P[u] = update(P[u],Q,u,i,r,k,rw,rc,alpha,lam)
        r_est = np.dot(P,Q.T)
        r_est[r_est<=0]=0
        r_est[r_est>=5]=5
        r_est[rc==0]=0
        mse = np.sqrt(np.sum((r_est-rc)**2)/np.sum(rc>0))
        print('Epoch:',epc,' RMSE:' , mse, ' change: ', old_mse-mse)
        old_mse = mse
        res_mse.append(mse)
    save_plot(res_mse,'training_varAlph.png')
    return P,res_mse

def save_plot(mse,f_name):
    path='./'+f_name
    x = np.arange(0,len(mse),1)
    fig = plt.figure()
    plt.plot(x,mse)
    plt.plot()
    fig.savefig(path,dpi=fig.dpi)

#%% Hyperparams
epochs = 200
lam = 0.05
lam_step = 20
lam_change = 0.001
k = 100 # number of components
alpha = 0.05
alpha_step = 25
alpha_change = 0.1
rmax = 5


#%% get_data
data,data_movies = util.get_user_movie_rating()
r  = np.asarray(data_movies,dtype=np.float32)[:,1:]
rc = np.asarray(data_movies[:1000],dtype=np.float32)[:,1:]
rw = np.asarray(data_movies[1000:],dtype=np.float32)[:,1:]

#%% computing Latent factors

# P = np.random.rand(rc.shape[0],k)
#P = np.load('500epc_p.npy')
P = get_P(rc,k)
Q = get_Q(r,k)

#%%
Rmse = []
P,Rmse = rapare(P,Q,rc,rw,rmax,epochs,k,alpha,lam,Rmse,lam_step,lam_change,alpha_step,alpha_change)

#np.save('try_1.38539_P.npy',P)
