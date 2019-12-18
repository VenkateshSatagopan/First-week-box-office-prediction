import numpy as np
import pandas as pd
import pickle
import os


def fix_date(x):
    """
    Fixes dates which are in 20xx
    """
    temp = x.split('/')
    if int(temp[0]) > 12:
        year = temp[0]
        return x[5:] + '/' + year
    else:
        year = x.split('/')[2]
        if int(year) < 1900:
            if int(year) <= 19:
                return x[:-2] + '20' + year
            else:
                return x[:-2] + '19' + year
        else:
            return x


class MDPred1:
    def __init__(self):
        self.CURRENT_MODEL_NAME = 'rf_score_first.pkl'
        self.rf_filename = 'rf_norevenue.pkl'
        self.mlb_filename = 'multiLabelBinarizer_rf.pkl'
        self.to_drop = ['original_title', 'imdb_id']
        self.estimated_statistics = {'budget': 21000000, 'popularity': 7.954623,
                                     'release_date': '01/01/98', 'genres': 'Drama'}

        model_folder = os.path.join(os.path.dirname(__file__), 'trained_models')

        rf_file = os.path.join(model_folder, self.rf_filename)
        mlb_file = os.path.join(model_folder, self.mlb_filename)

        self.rf_loaded = pickle.load(open(rf_file, 'rb'))
        self.mlb_loaded = pickle.load(open(mlb_file, 'rb'))

    def preprocessing_step(self, dataset_raw, estimated_statistics=None, to_drop=None, mlb=None):
        if estimated_statistics is None:
            estimated_statistics = self.estimated_statistics

        if to_drop is None:
            to_drop = self.to_drop

        if mlb is None:
            mlb = self.mlb_loaded

        ## drop 'wb_info', if there is:
        #try:
        #    del dataset['wb_info']
        #except:  # FIXME: too broad exception clause, do not use bare except
        #    pass
        dataset = dataset_raw.copy()
        # drop useless columns:
        for column in to_drop:
            del dataset[column]
        # add Boolean column 'has_collection'
        #bool_collection = ~pd.isnull(dataset.belongs_to_collection)
        
        
        # For the sake of test on AWS 
        # Don't have the following lines after retrain the model
        #dataset['has_collection'] = 1 #if bool_collection == -1 else 0
        #del dataset['belongs_to_collection']
        # add Boolean column 'has_homepage'
        #bool_homepage = ~pd.isnull(dataset.homepage)
        #dataset['has_homepage'] = 1 #if bool_homepage == -1 else 0
        #del dataset['homepage']
        
        
        # fix 'budget' values:
        #if dataset.budget.empty: 
        #    dataset.budget = estimated_statistics['budget']
        dataset.budget.fillna(value=self.estimated_statistics['budget'],inplace=True)   
        dataset.runtime.fillna(value=120.0,inplace=True)    
        # one-hot-encoding for genres
        genres_names = []
        if dataset.genres.empty: #pd.isnull(dataset.genres[0]):
            genres_names = [estimated_statistics['genres']]
        else:
            for d in dataset.genres.to_list():
                genres_names.append(d)
        x = mlb.transform(genres_names)
        genres_ohe = pd.Series(x[0], index=mlb.classes_)
        genres_ohe[genres_ohe.isnull()] = 0
        # dataset = pd.concat((dataset, genres_ohe), axis=0) # concate in the last of the function

        # add Boolean column 'num_genres'
        if dataset.genres.empty: #pd.isnull(dataset['genres'][0]):
            dataset['num_genres'] = 0
        else:
            dataset['num_genres'] = len(dataset['genres'][0])
        del dataset['genres']
        # add Boolean column 'num_companies'
        if dataset.production_companies.empty: #pd.isnull(dataset['production_companies'][0]):
            dataset['num_companies'] = 0
        else:
            dataset['num_companies'] = len(dataset['production_companies'][0])
        del dataset['production_companies']
        # add Boolean column 'num_countries'
        if dataset.production_countries.empty:#pd.isnull(dataset['production_countries'][0]):
            dataset['num_countries'] = 0
        else:
            dataset['num_countries'] = len(dataset['production_countries'][0])
        del dataset['production_countries']
        # add Boolean column 'num_languages'
        if dataset.spocken_languages.empty: #pd.isnull(dataset['spoken_languages'][0]):
            dataset['num_languages'] = 0
        else:
            dataset['num_languages'] = len(dataset['spocken_languages'][0])
        del dataset['spocken_languages']
        # add Boolean columns for genders of characters
        if dataset.cast.empty: #pd.isnull(dataset['cast'][0]):
            dataset['num_cast'] = 0
        else:
            dataset['num_cast'] = len(dataset['cast'][0])
        del dataset['cast']
        # add boolean column 'num_crew'
        if dataset.crew.empty: #pd.isnull(dataset['crew'][0]):
            dataset['num_crew'] = 0
        else:
            dataset['num_crew'] = len(dataset['crew'][0])
        del dataset['crew']
        # log-scale of some numerical variables with 'bad' distribution
        dataset.budget = np.log1p(int(dataset.budget.to_list()[0]))
        dataset.popularity.fillna(value=self.estimated_statistics['popularity'],inplace=True)
        dataset.popularity = np.log1p(int(dataset.popularity.to_list()[0]))
        # get release year
        if dataset.release_date.empty: #pd.isnull(dataset.release_date[0]):
            dataset.release_date = estimated_statistics['release_date']
        temp_date = dataset['release_date']
        dataset['release_date'] = int(temp_date[0][-4:])
        
        dataset = pd.concat((dataset.iloc[0], genres_ohe), axis=0)

        return dataset

    def __call__(self, input_vec):
        vec_preprocessed = self.preprocessing_step(input_vec)
        pred_result = self.rf_loaded.predict(vec_preprocessed.values.reshape((1, -1)))
        return pred_result
