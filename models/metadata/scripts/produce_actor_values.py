import os

import pandas as pd
import numpy as np

actors_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, os.pardir, 'name.basics.tsv'))  # noqa

actors = pd.read_csv(actors_dataset, sep = '\t') # read all actors, you need name.basics.tsv from IMDb

actors_clean = actors[['nconst', 'knownForTitles']] # keep only id and titles
actors_clean = actors_clean[actors_clean['knownForTitles']!=r'\N'] # cleanup, remove actors with no known movies

actors_clean

final_dataset = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, 'kaggle_data', 'Final_dataset.csv'))  # noqa
movies = pd.read_csv(final_dataset)

movies_clean = movies[['imdb_id', 'first_week_box_office']]

movies_clean

def value_from_ids(ids, movies_clean):
    actor_value = 0
    actor_movie_count = 0
    for movie in ids:
        bool_select = movies_clean.imdb_id == movie
        if sum(bool_select) == 1:
            actor_value += movies_clean[bool_select].first_week_box_office.to_list()[0]
            actor_movie_count + 1
            
    if actor_movie_count > 0:
        actor_value /= actor_movie_count
    return actor_value

for index, actor in actors_clean.iterrows():
    actor_movies = actor['knownForTitles'].split(',')
    actor_value = value_from_ids(actor_movies, movies_clean)
    actors_clean.iloc[index,:]['actor_values'] = actor_value

    if index%1000 ==0:
        print(index)

actors_clean['actor_values'] = actor_values

actors_clean[10]

