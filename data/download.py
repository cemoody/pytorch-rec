import pickle
import os.path
from zipfile import ZipFile
from functools import reduce

import requests
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

seed = 42
name = 'ml-20m.zip'
base = 'ml-20m'


def process_movies():
    movies = pd.read_csv(base + '/movies.csv', delimiter=',')
    # Genres column is pipe-seperated
    sets = (movies.genres
                  .apply(lambda x: set(x.split('|')))
                  .values)
    # Get unique genres
    genres = reduce(set.union, sets)
    # Clean zero-len genres
    genres = {genre for genre in genres
              if genre is not None and len(genre) > 0}

    # Build one-hot features
    genr_cols = []
    for index, genre in enumerate(sorted(genres)):
        col = 'is_' + genre.lower()
        movies[col] = movies.genres.apply(lambda x: genre in x) * index
        genr_cols.append(col)

    # Remove unused features
    del movies['title']
    del movies['genres']
    return movies, genr_cols


def process_ratings():
    # Download, unzip and read in the dataset
    if not os.path.exists(name):
        url = 'http://files.grouplens.org/datasets/movielens/' + name
        r = requests.get(url)
        with open(name, 'wb') as fh:
            fh.write(r.content)
        zipf = ZipFile(name)
        zipf.extractall()

    # First col is user, 2nd is movie id, 3rd is rating, 4th timestamp
    ratings = pd.read_csv(base + '/ratings.csv')
    movies, genr_cols = process_movies()
    ratings = ratings.merge(movies, on='movieId')
    return ratings, genr_cols


def extract():
    ratings, genr_cols = process_ratings()
    n_user = ratings.userId.max() + 1
    n_item = ratings.movieId.max() + 1
    n_genr = len(genr_cols)
    n_obs = len(ratings)

    # Note that we have a code / index that's shared over all movies, user,
    # and categorical features. So a single code corresponds to a single movie
    # or a single user or single feature. This is for compatibility with
    # factorization machines, which represent the input with a shared index
    # Zero in any case will correspond with empty or missing
    ratings['user_code'] = ratings.userId
    ratings['item_code'] = ratings.movieId + ratings.user_code.max() + 1
    genr_cols_code = [col + '_code' for col in genr_cols]
    im = ratings.item_code.max()
    for col, col_code in zip(genr_cols, genr_cols_code):
        # Map genre:0 to genre_code:0 but genre:4 to genre_code:offset + 4
        ratings[col_code] = ratings[col].map(lambda x: 0 if x == 0 else x + im)

    # These features are built for time-oriented training
    ratings['user_frame'] = (ratings.groupby('user_code')
                                    .timestamp
                                    .rank(method='dense', ascending=True)
                                    .astype('int'))
    ret = (ratings.groupby('user_code')
                  .user_frame
                  .transform(lambda x: np.ones_like(x) * x.max()))
    ratings['user_rank_n'] = ret

    # Build fast mappings between the codes and the user or movie Id
    code_to_key = {}
    for user_code, userId in zip(ratings.user_code, ratings.userId):
        code_to_key[user_code] = ('user', userId)
    for item_code, movieId in zip(ratings.item_code, ratings.movieId):
        code_to_key[item_code] = ('item', movieId)
    for col in genr_cols:
        for genr_code, genr in zip(ratings[col + '_code'], ratings[col]):
            code_to_key[genr_code] = ('genre', genr)
    key_to_code = {k: c for (c, k) in code_to_key.items()}
    np.savez('full', n_user=n_user, n_item=n_item, seed=seed, n_obs=n_obs,
             n_genr=n_genr)
    with open('code_to_key.pkl', 'wb') as fh:
        pickle.dump(code_to_key, fh)
    with open('key_to_code.pkl', 'wb') as fh:
        pickle.dump(key_to_code, fh)
    return ratings, genr_cols_code


def loocv():
    # Build train, test sets
    ratings, genr_cols_code = extract()
    train, test = train_test_split(ratings, random_state=seed)

    train_user = train.user_code.values
    train_item = train.item_code.values
    train_genr = train[genr_cols_code].values
    train_scor = train.rating.values

    test_user = test.user_code.values
    test_item = test.item_code.values
    test_genr = test[genr_cols_code].values
    test_scor = test.rating.values

    # One column for user id, item id, then a column
    # for each of the 18 genre ids
    # So our input will be of shape (n_obs, 20)
    # Note that the id space is unique over all
    train_featx = (train_user[:, None], train_item[:, None], train_genr)
    train_feat = np.concatenate(train_featx, axis=1)
    test_featx = (test_user[:, None], test_item[:, None], test_genr)
    test_feat = np.concatenate(test_featx, axis=1)

    # Save everything
    np.savez('loocv_train', train_feat=train_feat, train_scor=train_scor)
    np.savez('loocv_test', test_feat=test_feat, test_scor=test_scor)


def history():
    # Build train, test sets
    ratings, genr_cols_code = extract()

    # We have the dataframe as a series of events.
    # However, we need to re-orient this dataframe into
    # a format where a single row represents all events for a single
    # user. Subsequent events instead of rows will be placed into
    # subsequent columns.
    n_col = min(ratings.user_rank_n.max(), 1000)
    idx = ratings.user_frame < n_col
    sub = ratings.ix[idx]
    frac = (1.0 - idx.sum() * 1.0 / idx.shape[0]) * 100.0
    print(f"Removing {frac}% ratings from long history episodes ")
    shape = (sub.user_code.max() + 1, n_col)
    ratings_input = np.zeros(shape, dtype='int32')
    ratings_label = np.zeros(shape, dtype='float32')
    ratings_length = np.zeros(shape[0], dtype='int32')
    ratings_input[sub.user_code, sub.user_frame] = sub.item_code
    ratings_label[sub.user_code, sub.user_frame] = sub.rating
    ratings_length[sub.user_code] = sub.user_rank_n

    # Sort by ratings length so that we train with similarly-sized batches
    idx = np.argsort(ratings_length)
    ratings_length = ratings_length[idx]
    ratings_input = ratings_input[idx]
    ratings_label = ratings_label[idx]

    ret = train_test_split(ratings_length, ratings_input, ratings_label,
                           random_state=seed)
    (train_length, test_length, train_input, test_input,
     train_label, test_label) = ret

    # Save everything
    np.savez('history_train', train_length=train_length,
             train_input=train_input, train_label=train_label)
    np.savez('history_test', test_length=test_length,
             test_input=test_input, test_label=test_label)


if __name__ == '__main__':
    history()
    loocv()
