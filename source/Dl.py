import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reading ratings file
ratings = pd.read_csv('ratings.csv', sep='\t', encoding='latin-1', usecols=['user_id', 'movie_id', 'user_emb_id', 'movie_emb_id', 'rating'])
max_userid = ratings['user_id'].drop_duplicates().max()
max_movieid = ratings['movie_id'].drop_duplicates().max()

# Reading ratings file
users = pd.read_csv('users.csv', sep='\t', encoding='latin-1',usecols=['user_id', 'gender', 'zipcode', 'age_desc', 'occ_desc'])

# Reading ratings file
movies = pd.read_csv('movies.csv', sep='\t', encoding='latin-1', usecols=['movie_id', 'title', 'genres'])

print max_movieid
print max_userid

shuffled_ratings = ratings.sample(frac=1., random_state=0)

# Shuffling users
Users = shuffled_ratings['user_emb_id'].values
print 'Users:', Users, ', shape =', Users.shape

# Shuffling movies
Movies = shuffled_ratings['movie_emb_id'].values
print 'Movies:', Movies, ', shape =', Movies.shape

# Shuffling ratings
Ratings = shuffled_ratings['rating'].values
print 'Ratings:', Ratings, ', shape =', Ratings.shape


K_FACTORS = 100 # The number of dimensional embeddings for movies and users
TEST_USER = 2000 # A random test user (user_id = 2000)
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras.models import Model


inp1 = Input((1,))
x1 = Embedding(6040,100,input_length=1)(inp1)
x1 = Reshape((100,))(x1)
x1 = Reshape((100,))(x1)
inp2 = Input((1,))
x2 = Embedding(3952,100,input_length=1)(inp2)
x2 = Reshape((100,))(x2)
x3 = dot([x1,x2],axes=1)
model = Model([inp1,inp2],x3)
model.compile(loss='mse',optimizer='adamax')

callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint('weights.h5', save_best_only=True)]

# Use 30 epochs, 90% training data, 10% validation data 
history = model.fit([Users, Movies], Ratings, nb_epoch=30, validation_split=.1, verbose=1, callbacks=callbacks)