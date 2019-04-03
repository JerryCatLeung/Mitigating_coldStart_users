#%%
# Code for analysing and preprocessing of data

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sqlite3 as sql

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

#%%
data1= analysis_user()
data2= analysis_rating()
# plt.hist(data2)
# plt.plot(data1[:,0],data1[:,1])