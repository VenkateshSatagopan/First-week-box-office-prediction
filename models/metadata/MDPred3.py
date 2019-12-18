import os
import pandas as pd


class MDPred3:
    def __init__(self):
        print('Loading MDPred3')
        # find datasets
        data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, 'kaggle_data'))
        actors_data = os.path.abspath(os.path.join(data_folder, 'name.basics.tsv'))
        movies_data = os.path.abspath(os.path.join(data_folder, 'Final_dataset.csv'))

        # load actors
        actors = pd.read_csv(actors_data, sep='\t')  # read all actors, you need name.basics.tsv from IMDb
        actors_clean = actors[['nconst', 'knownForTitles']]  # keep only id and titles
        actors_clean = actors_clean[
            actors_clean['knownForTitles'] != r'\N']  # cleanup, remove actors with no known movies
        self.actors = actors_clean

        # load movies
        movies = pd.read_csv(movies_data)
        movies_clean = movies[['imdb_id', 'first_week_box_office']]
        self.movies = movies_clean
        print('Finished loading MDPred3')

    def actor_value_from_ids(self, ids):
        actor_value = 0
        actor_movie_count = 0
        for movie in ids:
            bool_select = self.movies.imdb_id == movie
            if sum(bool_select) == 1:
                actor_value += self.movies[bool_select].first_week_box_office.to_list()[0]
                actor_movie_count += 1

        if actor_movie_count > 0:
            actor_value /= actor_movie_count
        return actor_value

    def actor_value(self, actor_id):
        bool_vec = self.actors['nconst'] == actor_id
        if sum(bool_vec) == 1:
            actor = self.actors[bool_vec]
            actor_movies = actor.iloc[0]['knownForTitles'].split(',')
            actor_value = self.actor_value_from_ids(actor_movies)
            return actor_value
        else:
            return 0

    def __call__(self, actors, weights=None):
        if weights is None:
            weights = [1] * len(actors)
        print(actors)
        # only keep first actors
        weights = weights[0:14]
        actors = actors[0:14]

        actor_values = [self.actor_value(actor) for actor in actors]

        movie_value = 0
        for index, actor_value in enumerate(actor_values):
            movie_value += actor_value * weights[index]

        if len(weights) > 0:
            movie_value /= sum(weights)

        return movie_value


if __name__ == "__main__":
    m = MDPred3()
    val = m(['nm0000375', 'nm0262635'])
    print(val)