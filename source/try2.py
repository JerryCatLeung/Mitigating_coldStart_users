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
from MF import ProductRecommender

# %% Helper-Functions
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
rc_train = np.asarray(data_movies[:900],dtype=np.float32)[:,1:]
rc_test = np.asarray(data_movies[900:1000],dtype=np.float32)[:,1:]
rw = np.asarray(data_movies[1000:],dtype=np.float32)[:,1:]

#%%
model = ProductRecommender()
model.fit(rc_train)
#%% computing Latent factors

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
