#%%
# Code for analysing and preprocessing of data
#%%
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import sqlite3 as sql
import ast
from scipy.linalg import svd
#%%
def get_data(f_name):
    movies = pd.read_csv(f_name+'movies.csv')
    users = pd.read_csv(f_name+'users.csv')
    ratings = pd.read_csv(f_name+'ratings.csv')

    return movies,users,ratings

def create_movies_table(df):
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("SELECT count(*) from sqlite_master where type = 'table' and name = 'movies_data';")
    flag = cur.fetchall()[0][0]

    if flag == 0 :
        cur.execute("CREATE TABLE if not exists movies_data (M_id INT PRIMARY KEY, Title TEXT, Genres TEXT);")
        df = df[['movie_id','title','genres']]

        for id,rows in df.iterrows():
            cur.execute("INSERT INTO movies_data(M_id, Title, Genres) values(?,?,?);",rows)
        db.commit()
        print ("movies_data table was created with entries")

    else :
        print ("movies_data table already exists")
    db.close()

def create_users_table(df):
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("SELECT count(*) from sqlite_master where type = 'table' and name = 'users_data';")
    flag = cur.fetchall()[0][0]

    if flag == 0 :
        cur.execute("CREATE TABLE if not exists users_data (U_id INT PRIMARY KEY, Gender TEXT, Age INT);")
        df = df[['user_id','gender','age']]

        for id,rows in df.iterrows():
            cur.execute("INSERT INTO users_data(U_id, Gender, Age) values(?,?,?);",rows)
        db.commit()
        print ("users_data table was created with entries")

    else :
        print ("users_data table already exists")
    db.close()

def create_ratings_table(df):
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("SELECT count(*) from sqlite_master where type = 'table' and name = 'ratings_data';")
    flag = cur.fetchall()[0][0]

    if flag == 0 :
        cur.execute("CREATE TABLE if not exists ratings_data (U_id INT, M_id INT, rating INT);")
        df = df[['user_id','movie_id','rating']]

        for id,rows in df.iterrows():
            cur.execute("INSERT INTO ratings_data(U_id, M_id, rating) values(?,?,?);",rows)
        db.commit()
        print ("ratings_data table was created with entries")

    else :
        print ("ratings_data table already exists")
    db.close()

def Drop_table(t_name):
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("DROP TABLE if exists "+t_name +";")
    db.close()
#%%

movies,users,ratings = get_data('../Data/')

#%%
print(movies.head(10))
print(movies.shape)
print(users.head(10))
print(users.shape)
print(ratings.head(10))
print(ratings.shape)

#%%
create_movies_table(movies)
create_users_table(users)
create_ratings_table(ratings)
#%%

def analysis_user():
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("Select cnt,count(*) as num_users from (select U_id,count(*) as cnt from ratings_data group by U_id order by cnt ) group by cnt")
    data1 = cur.fetchall()
    db.close()
    return np.asarray(data1)

def analysis_rating():
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("select rating from ratings_data")
    data2 = cur.fetchall()
    db.close()
    return np.asarray(data2)


def get_user_movie_rating():
    db = sql.connect(os.path.abspath('../Data/Movie.db'))
    cur = db.cursor()
    cur.execute("select U_id,'{'||group_concat(M_id|| ':' || rating , ',')||'}' as movie_list from ratings_data group by U_id limit 6040;")
    data = cur.fetchall()
    db.close()
    dict = {int(data[i][0]):ast.literal_eval(data[i][1]) for i in range (len(data))}

    data_movies = np.full((6041,3953),0)
    for keys in dict:
        data_movies[keys][0] = len(dict[keys])
        for k in dict[keys]:
            data_movies[keys][k] = dict[keys][k]

    data_movies = sorted(data_movies, key= lambda entry : entry[0])
    return dict,data_movies

#%%
data,data_movies = get_user_movie_rating()
cold_users = data_movies[:1000]
warm_users = data_movies[1000:]
# cold_users[1]
R = np.asarray(cold_users,dtype='float64')
cold_users = R[:,1:]
R = R[:,1:]
# user_ratings_mean = np.mean(R, axis = 1)
# R_demeaned = R - user_ratings_mean.reshape(-1, 1)
from scipy.sparse.linalg import svds
U,sigma,_ = svds(R, k = 50)
P = np.dot(U,np.diag(sigma))


R = np.asarray(warm_users,dtype='float64')
warm_users = R[:,1:]
R = R[:,1:]
#user_ratings_mean = np.mean(R, axis = 1)
#R_demeaned = R - user_ratings_mean.reshape(-1, 1)
_, _, Q = svds(R, k = 50)

Q=Q.T

lam,alpha = 1,0.01 #change it please

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

def update(p,i,j,r,f , count_r):
    rui = cold_users[i][j]
    P_ = np.reshape(P[i][:],(-1,1)).T
    Q_ = np.reshape(Q[:][j],(-1,1))
    # print (P_.shape,Q_.shape)
    rcap = np.dot(P_,Q_)
    grad = 2 * Q[i][f] * count_r * (g_linear(rui,r) - g_linear(rcap,r)) + (2 * lam * P[i][f])
    return (p - alpha * grad)

def func(cold_users,P,Q,r_max = 5,k=50):
    count_r = np.empty(( r_max , len(cold_users[0])))
    for r in range(r_max):
        for j in range(len(cold_users)):
            count_r[r][j] = np.count_nonzero(warm_users[:][j]==r)
    for temp in range(2):
        print(temp)
        for i in range(len(cold_users)):
            for j in range(len(cold_users[0])):
                if (cold_users[i][j] != 0):
                    for r in range(r_max):
                        for f in range(k):
                            P[i][f] = update(P[i][f],i,j,r,f , count_r[r][j])
    return P
def SVD():
    R = np.asarray(data_movies, dtype='float64')
    R = R[:, 1:]
    user_ratings_mean = np.mean(R, axis = 1)
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)
    U, sigma, Vt = svds(R_demeaned, k=50)
    sigma_d = np.diag(sigma)
    predicted = np.dot(np.dot(U, sigma_d), Vt) + user_ratings_mean.reshape(-1, 1)
    return predicted

def MF():
    R = np.asarray(data_movies, dtype='float64')
    R = R[:, 1:]
    MF_K = 10
    MF_P = np.random.normal(scale=1. / MF_K, size=(np.size(R , 0), MF_K))
    MF_Q = np.random.normal(scale=1. / MF_K, size=(np.size(R , 1), MF_K))
    b_u = np.zeros(np.size(R , 0))
    b_i = np.zeros(np.size(R , 1))
    b = np.mean(R[np.where(R != 0)])
    samples = [
        (i, j, R[i, j])
        for i in range(np.size(R , 0))
        for j in range(np.size(R , 1))
        if R[i, j] > 0
    ]
    iterations = 10
    for i in range(iterations):
        np.random.shuffle(samples)
        for i, j, r in samples:
            # Computer prediction and error
            prediction = b + b_u[i] + b_i[j] + MF_P[i, :].dot(MF_Q[j, :].T)
            e = (r - prediction)
            beta = 0.1
            # Update biases
            b_u[i] += alpha * (e - beta * b_u[i])
            b_i[j] += alpha * (e - beta * b_i[j])

            # Update user and item latent feature matrices
            MF_P[i, :] += alpha * (e * MF_Q[j, :] - beta * MF_P[i, :])
            MF_Q[j, :] += alpha * (e * MF_P[i, :] - beta * MF_Q[j, :])
        xs, ys = R.nonzero()
        predicted = b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + MF_P.dot(MF_Q.T)
        error = 0
        cnt = 0
        for x, y in zip(xs, ys):
            cnt = cnt + 1
            error += pow(R[x, y] - predicted[x, y], 2)
        print(np.sqrt(error / cnt))
    xs, ys = R.nonzero()
    predicted = b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + MF_P.dot(MF_Q.T)
    error = 0
    cnt = 0
    for x, y in zip(xs, ys):
        cnt = cnt + 1
        error += pow(R[x, y] - predicted[x, y], 2)
    print(np.sqrt(error / cnt))
    return b + b_u[:, np.newaxis] + b_i[np.newaxis:, ] + MF_P.dot(MF_Q.T)

#P = func(cold_users,P,Q)
predicted= SVD()
R = np.asarray(data_movies, dtype='float64')
R = R[:, 1:]
xs, ys = R.nonzero()
error = 0
cnt = 0
for x, y in zip(xs, ys):
    cnt = cnt + 1
    error += pow(R[x, y] - predicted[x, y], 2)
print(np.sqrt(error / cnt))
#print(P.shape)
print(P2.shape)
print(P)
#print(P2)
