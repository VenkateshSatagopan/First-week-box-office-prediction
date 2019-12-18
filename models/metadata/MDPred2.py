import os
from keras.models import load_model
import pickle
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
import numpy as np
import pandas as pd


class MDPred2:
    def __init__(self):
        self.model_name_1='xgb_model_full.dat'
        self.model_name_2='catmodel.dat'
        self.model_name_3 = 'lgb_model.dat'
        self.model_name_4='ensembled_result.h5'
        self.to_drop = ['release_date', 'id', 'genres', 'homepage', 'original_title', 'title', 'original_language',
                        'overview', 'production_companies', 'production_countries', 'spoken_languages', 'status',
                        'tagline', 'belongs_to_collection', 'imdb_id', 'poster_path', 'Keywords', 'cast', 'crew']

        model_folder = os.path.join(os.path.dirname(__file__), 'trained_models')
        model_file_1 = os.path.join(model_folder, self.model_name_1)
        model_file_2 = os.path.join(model_folder, self.model_name_2)
        model_file_3 = os.path.join(model_folder, self.model_name_3)
        model_file_4 = os.path.join(model_folder, self.model_name_4)
        self.model_1=  pickle.load(open(model_file_1, 'rb'))
        self.model_2=  pickle.load(open(model_file_2, 'rb'))
        self.model_3 = pickle.load(open(model_file_3, 'rb'))
        #self.model_4 = load_model(model_file_4)

        self.estimated_statistics = {'budget': 26494751.897649363 , 'popularity': 7.954623, 
                        'release_date':'01/01/98', 'genres': 'Drama'}

    def drop(self, test):
        # Drop specified attributes from the training and test data sets
        for x in self.to_drop:
            try:
                if x != 'revenue' or x != 'wb_info' or x != 'first_week_box_office':
                    test = test.drop(x, axis=1)
            except:  # FIXME: too broad exception clause, do not use bare except
                pass
        return test
    def lazzy_feat(self,df=None):
    
        df['Ratiobudgetbypopularity'] = df['budget']/df['popularity']
        df['RatiopopularitybyYear'] = df['popularity']/df['Year']
        df['RatoioruntimebyYear'] = float(df['runtime'])/df['Year']
    
    
        df['budget_runtime_ratio'] = df['budget']/float(df['runtime']) 
        df['budget_Year_ratio'] = df['budget']/df['Year']
    
        return df

    def pre_processing(self,df=None,estimated_statistics=None):
        #if pd.isna(df['budget']):
        #  df['budget'] = self.estimated_statistics['budget']
        #if pd.isna(df['popularity']):
        #  df['popularity'] = self.estimated_statistics['popularity']
        df['budget'].fillna(value=self.estimated_statistics['budget'], inplace=True)
        df['runtime'].fillna(value=120.0, inplace=True)
        df['popularity'].fillna(value=self.estimated_statistics['popularity'], inplace=True)
        df['budget'] = np.log1p(float(df['budget']))
        df['popularity'] = np.log1p(float(df['popularity']))
        df['nb_spoken_languages'] = df.spocken_languages.apply(len)
        df['english_spoken'] = df.spocken_languages.apply(lambda x: 'en' in x)
        main_languages=['en', 'fr', 'hi', 'ja', 'es', 'zh', 'de', 'it', 'ko', 'cn']
        dict_language = dict(zip(main_languages, range(1, len(main_languages)+1)))
        dict_language['other'] = 0
        list_spoken_languages=list(df['spocken_languages'])
        #df['original_language']=df['spocken_languages'][0]
        if len(list_spoken_languages)==1:
            df['original_language']=df['spocken_languages']
        else:
    	    df['original_language']=df['spocken_languages'][0]

        df.original_language = df.original_language.apply(lambda x: x if x in main_languages else 'other')
        temp_test = df[['imdb_id','genres']]
        genres = ['Comedy',
         'Drama',
         'Family',
         'Romance',
         'Horror',
         'Thriller',
         'Documentary',
         'Action',
         'Music',
         'Adventure',
         'Crime',
         'Science Fiction',
         'Mystery',
         'Foreign',
         'Animation',
         'Fantasy',
         'War',
         'Western',
         'History']
        for g in genres:
          temp_test[g] = temp_test.genres.apply(lambda x: 1 if g in x else 0)
        X_test=temp_test.drop(['genres', 'imdb_id'], axis=1).values
        # Number of features we want for genres
        n_comp_genres = 3

        # Build the SVD pipeline
        svd = make_pipeline(
        TruncatedSVD(n_components=n_comp_genres),
        Normalizer(norm='l2', copy=False)
        )
        f_test = svd.fit_transform(X_test)
        df = pd.concat([df, temp_test.iloc[:,1:]], axis=1)
        list_of_companies = list(df['production_companies'].apply(lambda x: [i['name'] for i in x] if x != {} else []).values)
        df.production_companies=df.production_companies.apply(lambda x: [i['name'] for i in x] if x != {} else []).values
        df['nb_production_companies'] = df.production_companies.apply(len)
        big_companies=['Paramount Pictures','United Artists','Metro-Goldwyn-Mayer (MGM)','Walt Disney Pictures','Castle Rock Entertainment','Twentieth Century Fox Film Corporation','DreamWorks SKG','Amblin Entertainment','BBC Films','UK Film Council','Canal+','New Line Cinema','Universal Pictures','Summit Entertainment','Warner Bros.','Columbia Pictures Corporation','Lakeshore Entertainment','The Weinstein Company','TriStar Pictures','Columbia Pictures','Hollywood Pictures','PolyGram Filmed Entertainment','Imagine Entertainment','Touchstone Pictures','Orion Pictures','Epsilon Motion Pictures','Regency Enterprises','Miramax Films',
                        'Ciné+',
                        'Revolution Studios',
                        'Silver Pictures',
                        'Fox Searchlight Pictures',
                        'StudioCanal',
                        'Legendary Pictures',
                        'Morgan Creek Productions',
                        'Village Roadshow Pictures',
                        'Screen Gems',
                        'Original Film',
                        'Lionsgate',
                        'Dimension Films',
                        'Dune Entertainment',
                        'New Regency Pictures',
                        'Relativity Media',
                        'Millennium Films',
                        'Working Title Films',
                        'Film4',
                        'Fox 2000 Pictures',
                        'France 2 Cinéma',
                        'Spyglass Entertainment',
                        'Davis Entertainment',
                        'Scott Rudin Productions',
                        'Participant Media',
                        'Focus Features',
                        'A24',
                        'Scott Free Productions',
                        'Wild Bunch',
                        'DreamWorks Animation',
                        'Lions Gate Films',
                        'TSG Entertainment']
        df.production_companies = df.production_companies.apply(lambda l: list(map(lambda x: x if x in big_companies else 'other', l)))
        temp_test_1 = df[['imdb_id', 'production_companies']]
        #temp_test = test[['id', 'production_companies']]

        for p in big_companies + ['other']:
          temp_test_1[p] = temp_test_1.production_companies.apply(lambda x: 1 if p in x else 0)
            #temp_test[p] = temp_test.production_companies.apply(lambda x: 1 if p in x else 0)
    
        X_test_1 = temp_test_1.drop(['production_companies', 'imdb_id'], axis=1).values
        #X_test = temp_test.drop(['production_companies', 'id'], axis=1).values

        # Number of features we want for genres
        n_comp_production_companies = 3

        # Build the SVD pipeline
        svd = make_pipeline(TruncatedSVD(n_components=n_comp_production_companies),
                            Normalizer(norm='l2', copy=False))

        # Here are our new features
        f_test_1 = svd.fit_transform(X_test_1)
        #print(f_test.shape)
        if len(f_test_1)<n_comp_production_companies:
            pad_val_1=np.zeros((1,n_comp_production_companies-len(f_test_1)))
            f_test_1=np.concatenate((f_test_1, pad_val_1),axis=1)
            #print(pad_val.shape)
        for i in range(n_comp_production_companies):
            df['production_companies_reduced_{}'.format(i)] = f_test_1[:, i]
        big_countries=['US','CA',
                   'GB', 'DE',
                   'FR',
                   'NZ',
                   'JP',
                   'IE',
                   'IT',
                   'CN',
                   'HK',
                   'AU',
                   'SE',
                   'NL',
                   'IN',
                   'ES',
                   'BE',
                   'MX',
                   'DK']
        df.production_countries = df.production_countries.apply(lambda l: list(map(lambda x: x if x in big_countries else 'other', l)))

        temp_test_2 = df[['imdb_id', 'production_countries']]
        #temp_test = test[['id', 'production_countries']]

        for p in big_countries + ['other']:
            temp_test_2[p] = temp_test_2.production_countries.apply(lambda x: 1 if p in x else 0)
            #temp_test[p] = temp_test.production_countries.apply(lambda x: 1 if p in x else 0)
    
        X_test_2 = temp_test_2.drop(['production_countries', 'imdb_id'], axis=1).values
        #X_test = temp_test.drop(['production_countries', 'id'], axis=1).values

        # Number of features we want for genres
        n_comp_production_countries = 3

        # Build the SVD pipeline
        svd = make_pipeline(TruncatedSVD(n_components=n_comp_production_countries),
                        Normalizer(norm='l2', copy=False))

        # Here are our new features
        f_test_2 = svd.fit_transform(X_test_2)
        if len(f_test_2)<n_comp_production_countries:
            pad_val_2=np.zeros((1,n_comp_production_countries-len(f_test_2)))
            f_test_2=np.concatenate((f_test_2, pad_val_2),axis=1)
            #print(pad_val.shape)
        for i in range(n_comp_production_countries):
            df['production_countries_reduced_{}'.format(i)] = f_test_2[:, i]
        df.release_date.fillna('05/05/05')
        df['release_date'] = pd.to_datetime(df['release_date'])
        print(df.release_date)
        df['Year'] = df.release_date.dt.year
        df['Month'] = df.release_date.dt.month
        df['Day'] = df.release_date.dt.day
        df['dayofweek'] = df.release_date.dt.dayofweek 
        df['quarter'] = df.release_date.dt.quarter   
        for i in range(2,13):
            df['Month'+str(i)]=0
        month_val=float(df['Month'])
        if not month_val==1:
         df['Month'+str(month_val)]=1
        for i in range(1,7):
            df['dayofweek'+str(i)]=0

        week_val=float(df['dayofweek'])
        #week_val=week_val-1
        if not week_val==0:
            df['dayofweek'+str(week_val)]=1
        print(df['Month'])
        print(df['Day'])
        #dummies = pd.get_dummies(df['Month'] ,drop_first=True).rename(columns=lambda x: 'Month' + str(x))
        #dummies2 = pd.get_dummies(test['Month'] ,drop_first=True).rename(columns=lambda x: 'Month' + str(int(x)))
        #df = pd.concat([df, dummies], axis=1)
        #test = pd.concat([test, dummies2], axis = 1)
        #ddow = pd.get_dummies(df['dayofweek'] ,drop_first=True).rename(columns=lambda x: 'dayofweek' + str(x))
        #ddow2 = pd.get_dummies(test['dayofweek'] ,drop_first=True).rename(columns=lambda x: 'dayofweek' + str(int(x)))
        #df = pd.concat([df, ddow], axis=1)

        df.loc[df['Year'] > 2019, 'Year'] = df.loc[df['Year'] > 2019, 'Year'].apply(lambda x: x - 100)
        df['runtime']=float(df['runtime'])
        train=df
        df=self.lazzy_feat(train)
        lst2=['Samuel L. Jackson', 'Robert De Niro', 'Morgan Freeman', 'Bruce Willis', 'Liam Neeson', 'Steve Buscemi', 'Willem Dafoe', 'Nicolas Cage', 'Matt Damon', 'John Goodman', 'Robin Williams', 'J.K. Simmons', 'Julianne Moore', 'Johnny Depp', 'Stanley Tucci', 'Richard Jenkins', 'Ben Stiller', 'Christopher Walken', 'Tom Hanks', 'Brad Pitt', 'John Leguizamo', 'Bill Murray', 'Sylvester Stallone', 'James Franco', 'Keith David', 'Woody Harrelson', 'Dennis Quaid', 'Owen Wilson', 'Paul Giamatti', 'Ben Affleck', 'Susan Sarandon', 'Tom Cruise', 'Forest Whitaker', 'John Turturro', 'Ben Kingsley', 'Alec Baldwin', 'Frank Welker', 'Kevin Costner', 'Dustin Hoffman', 'Denzel Washington', 'William H. Macy', 'Robert Downey Jr.', 'Ewan McGregor', 'Kevin Bacon', 'Antonio Banderas', 'John Cusack', 'Brendan Gleeson', 'Harrison Ford', 'Michael Shannon', 'Julia Roberts', 'John C. Reilly', 'Danny Glover', 'Nicole Kidman', 'Dan Aykroyd', 'Justin Long', 'Sigourney Weaver', 'Philip Seymour Hoffman', 'Brian Cox', 'Michael Caine', 'Stephen Root', 'Arnold Schwarzenegger', 'Laurence Fishburne', 'Eddie Murphy', 'Ethan Hawke', 'Tommy Lee Jones', 'Kevin Spacey', 'John Travolta', 'Danny DeVito', 'Cameron Diaz', 'John Hurt', 'Meryl Streep', 'Tom Wilkinson', 'Cate Blanchett', 'Gwyneth Paltrow', 'Bruce McGill', 'Matthew McConaughey', 'Harvey Keitel', 'Bill Nighy', 'Gary Oldman', 'Donald Sutherland', 'George Clooney', 'Robert Duvall', 'Kathy Bates', 'Scarlett Johansson', 'Octavia Spencer', 'Will Ferrell', 'Elizabeth Banks', 'John Cleese', 'Jim Broadbent', 'Steve Zahn', 'Danny Trejo', 'Luis Guzm√°n', 'John Malkovich', 'Mark Wahlberg', 'Bill Hader', 'Allison Janney', 'Vince Vaughn', 'Channing Tatum', 'James Earl Jones', 'Naomi Watts', 'Ed Harris', 'Ray Liotta', 'Gene Hackman', 'Oliver Platt', 'Jack Black', 'Seth Rogen', 'Stellan Skarsg√•rd', 'Christian Slater', 'James Cromwell', 'Whoopi Goldberg', 'Mel Gibson', 'Bill Paxton', 'Adam Sandler', 'Jonah Hill', 'Ciar√°n Hinds', 'Jason Bateman', 'Mark Ruffalo', 'Jude Law', 'Drew Barrymore', 'Kirsten Dunst', 'Anthony Mackie', 'Jennifer Aniston', 'Paul Rudd', 'Anthony Hopkins', 'Chris Cooper', 'Dwayne Johnson', 'Ralph Fiennes', 'Jason Statham', 'Philip Baker Hall', 'Keanu Reeves', 'Richard Gere', 'Don Cheadle', 'Christopher Plummer', 'Billy Bob Thornton', 'Halle Berry', 'James Marsden', 'Alfred Molina', 'Nick Nolte', 'Kristen Stewart', 'David Koechner', 'Michael Gambon', 'Dennis Hopper', 'James Remar', 'Andy Garc√≠a', 'Val Kilmer', 'Mark Strong', 'Natalie Portman', 'Colin Farrell', 'Peter Stormare', 'Elijah Wood', 'Carla Gugino', 'Helen Mirren', 'Dan Hedaya', 'Sean Penn', 'Philip Ettington', 'Beth Grant', 'Josh Brolin', 'Catherine Keener', 'Jeffrey Tambor', 'Rachel Weisz', 'Bradley Cooper', 'Liev Schreiber', 'Alan Rickman', 'Joan Cusack', 'Terrence Howard', 'Queen Latifah', 'Patrick Stewart', 'Ving Rhames', 'John Lithgow', 'Bill Pullman', 'Viola Davis', 'Stephen Tobolowsky', 'Jeremy Piven', 'Charlie Sheen', 'Bruce Greenwood', 'Michael Douglas', 'Jim Carrey', 'Robert Patrick', 'Emma Thompson', 'Rosario Dawson', 'Alan Arkin', 'Kristen Wiig', 'Steve Carell', 'Stan Lee', 'Charlize Theron', 'Ryan Reynolds', 'William Hurt', 'Christopher Lloyd', 'Dermot Mulroney', 'Judy Greer', 'Joseph Gordon-Levitt', 'Mickey Rourke', 'Colleen Camp', 'Jon Voight', 'Geoffrey Rush', 'Patricia Clarkson', 'Toby Jones', 'Angelina Jolie', 'Ron Perlman', 'Anna Faris', 'Christian Bale', 'David Thewlis', 'Helena Bonham Carter', 'Timothy Spall', 'Harry Dean Stanton', 'Anjelica Huston', 'Sam Rockwell', 'Eugene Levy', 'Steve Martin', 'Amy Adams', 'Alan Tudyk', 'Kurt Russell', 'Jeffrey Wright', 'Joe Pantoliano', 'Sean Bean', 'Giovanni Ribisi', 'Michael Pe√±a', 'Hugh Jackman', 'Tom Sizemore', "Vincent D'Onofrio", 'Seth Green', 'Chris Rock', 'Sandra Bullock', 'Jeff Daniels', 'Kate Winslet', 'Eddie Marsan', 'Al Pacino', 'Wallace Shawn', 'Michael Sheen', 'Guy Pearce', 'Pierce Brosnan', 'Christopher McDonald', 'Jared Harris', 'Jackie Chan', 'Ice Cube', 'Judi Dench', 'Elias Koteas', 'Tim Robbins', 'Steve Coogan', 'Luke Wilson', 'Benicio del Toro', 'Kevin Dunn', 'David Strathairn', 'Jeff Bridges', 'Maggie Smith', 'Will Smith', 'Diane Lane', 'Clint Howard', 'Michael Keaton', 'Martin Sheen', 'Adam Scott', 'James Woods', 'Tim Blake Nelson', 'Jason Flemyng', 'Cuba Gooding Jr.', 'Keira Knightley', 'Peter Sarsgaard', 'Laura Linney', 'Michael Fassbender', 'Russell Crowe', 'Jennifer Connelly', 'Jamie Foxx', 'Jon Favreau', 'Idris Elba', 'Angela Bassett', 'Edward Norton', 'John Candy', 'John Ratzenberger', 'Kiefer Sutherland', 'Sam Shepard', 'Jake Gyllenhaal', 'Marcia Gay Harden', 'Cheech Marin', 'Salma Hayek', 'Aaron Eckhart', 'Pen√©lope Cruz', 'Jeremy Irons', 'Kevin Hart', 'John Carroll Lynch', 'Michelle Pfeiffer', 'Clancy Brown', 'Kevin Pollak', 'Bob Hoskins', 'John Michael Higgins', 'Jane Lynch', 'Michael McKean', 'Jennifer Coolidge', 'James Hong', 'Uma Thurman', 'Amanda Peet', 'Vin Diesel', 'Margo Martindale', 'Daniel Craig', 'James Caan', 'Matthew Broderick', 'Viggo Mortensen', 'Chlo√´ Grace Moretz', 'Danny McBride', 'Hank Azaria', 'Kim Basinger', 'Winona Ryder', 'Chris Evans', 'Tilda Swinton', 'Jennifer Lopez', 'Sharon Stone', 'Brian Dennehy', 'Anna Kendrick', 'Jeff Goldblum', 'William Fichtner', 'Robbie Coltrane', 'Meg Ryan', 'Djimon Hounsou', 'Maria Bello', 'Jim Cummings', 'Jason Isaacs', 'Will Patton', 'Xander Berkeley', 'Wesley Snipes', 'Jonathan Pryce', 'Jennifer Garner', 'Carrie Fisher', 'Dylan Baker', 'Giancarlo Esposito', 'Cedric the Entertainer', 'Toni Collette', 'Kate Hudson', 'Diane Keaton', 'James Gandolfini', 'Frankie Faison', 'Paul Bettany', 'Ian McKellen', 'Hugo Weaving', 'Marisa Tomei', 'Greg Kinnear', 'Jim Belushi', 'Terry Crews', 'Thomas Lennon', 'Joe Chrest', 'Kathryn Hahn', 'Demi Moore', 'Zoe Saldana', 'Molly Shannon', 'Fred Ward', 'Tony Shalhoub', 'Miriam Margolyes', 'Jessica Alba', 'Common', 'Cary Elwes', 'Maya Rudolph', 'Frances McDormand', 'Mary Steenburgen', 'Ken Jeong', 'Rose Byrne', 'Terence Stamp', 'Ren√©e Zellweger', 'Brad Dourif', 'Lin Shaye', 'Dennis Haysbert', 'Kristen Bell', 'Jeremy Renner', 'Gerard Butler', 'Rob Corddry', 'Anne Hathaway', 'H√©ctor Elizondo', 'Bradley Whitford', 'Heather Graham', 'Seann William Scott', 'Laura Dern', 'Cliff Curtis', 'Kevin Kline', 'Eva Mendes', 'John Hawkes', 'John C. McGinley', 'Lance Henriksen', 'Juliette Lewis', 'Simon Pegg', 'David Jensen', 'John Ortiz', 'Rob Schneider', 'Frank Langella', 'Ashley Judd', 'Barry Pepper', 'Emma Stone', 'Bob Gunton', 'Richard Schiff', 'Hugh Grant', 'Timothy Olyphant', 'Sean Connery', 'Allen Covert', 'Glenn Morshower', 'Udo Kier', 'Danny Huston', 'Ben Foster', 'Charles Dance', 'Casey Affleck', 'John Krasinski', 'Christina Ricci', 'Patton Oswalt', 'Bob Balaban', 'Max von Sydow', 'Rachel McAdams', 'Will Arnett', 'Michael Madsen', 'Malcolm McDowell', 'Jean-Claude Van Damme', 'Burt Reynolds', 'Brendan Fraser', 'Ian Holm', 'Glenn Close', 'Leonardo DiCaprio', 'Emily Blunt', 'Ted Levine', 'Shea Whigham', 'Siobhan Fallon', 'David Morse', 'Pruitt Taylor Vince', 'Tom Kenny', 'Martin Short', 'Sam Elliott', 'Bobby Cannavale', 'J.D. Evermore', "Denis O'Hare", 'David Paymer', 'Jason Sudeikis', 'Olivia Wilde', 'John Heard', 'Joaquin Phoenix', 'Dean Norris', 'Sam Neill', 'Clive Owen', 'David Cross', 'Christina Applegate', 'Billy Crudup', 'Reese Witherspoon', 'Scott Glenn', 'Celia Weston', 'Jason Lee', 'Shia LaBeouf', 'Robert Loggia', 'Tim Curry', "Catherine O'Hara", 'John Cho', 'Debi Mazar', 'Jess Harnell', 'Mike Epps', 'Kris Kristofferson', 'R. Lee Ermey', 'Josh Lucas', 'Miranda Richardson', 'Christopher Lee', 'Anthony Anderson', 'Charles S. Dutton', 'Zach Galifianakis', 'Michelle Monaghan', 'Famke Janssen', 'Colin Firth', 'Rhys Ifans', 'Milla Jovovich', 'Robert Redford', 'Fred Tatasciore', 'Amy Poehler', 'John Diehl', 'Michael Clarke Duncan', 'Robin Wright', 'Woody Allen', 'James McAvoy', 'Paul Dano', 'Elle Fanning', 'Jesse Eisenberg', 'Richard Riehle', 'Toby Kebbell', 'Michael Wincott', 'Kathy Baker', 'Bryan Cranston', 'Janeane Garofalo', 'Kevin Corrigan', 'Leslie Mann', 'M. Emmet Walsh', 'Emily Mortimer', 'Randy Quaid', 'Clint Eastwood', 'Warwick Davis', 'Ryan Gosling', 'Clifton Collins Jr', 'Michael Rooker', 'Cloris Leachman', 'Patrick Warburton', 'Chiwetel Ejiofor', 'Chris Pine', 'Parker Posey', 'Fred Willard', 'Rachael Harris', 'Annette Bening', 'Kristin Scott Thomas', 'Tom Hollander', 'Rene Russo', 'Benjamin Bratt', 'Adrien Brody', 'Mary Ellen Trainor', 'Jessica Chastain']
        for temp in df['cast']:
            myvalues = [i['name'] for i in temp if 'name' in i]
            pop_actor=list(set(lst2).intersection(set(myvalues)))
            df['cast_score']=len(pop_actor)
        lst2_crew=['Avy Kaufman', 'Steven Spielberg', 'Robert Rodriguez', 'Mary Vernieu', 'Deborah Aquila', 'Bob Weinstein', 'Hans Zimmer', 'Harvey Weinstein', 'James Newton Howard', 'Tricia Wood', 'James Horner', 'Francine Maisler', 'Kerry Barden', 'Alan Silvestri', 'Danny Elfman', 'John Williams', 'Billy Hopkins', 'Suzanne Smith', "Dan O'Connell", 'Luc Besson', 'Jerry Goldsmith', 'Arnon Milchan', 'Scott Rudin', 'John Papsidera', 'Marco Beltrami', 'Mark Isham', 'Bruce Berman', 'John Debney', 'Ridley Scott', 'Brian Grazer', 'Joel Silver', 'Christophe Beck', 'Denise Chamian', 'Graeme Revell', 'Tim Bevan', 'Clint Eastwood', 'Kevin Kaska', 'Eric Fellner', 'Carter Burwell', 'Jane Jenkins', 'Stan Lee', 'Sarah Finn', 'Steven Soderbergh', 'Janet Hirshenson', 'Christopher Young', 'John Hughes', 'Mindy Marin', 'Hans Bjerno', 'John Powell', 'Bill Corso', 'John T. Cucci', 'Debra Zane', 'Ve Neill', 'Howard Shore', 'Joel Coen', 'J.J. Makaro', 'Thomas Newman', 'Ronna Kress', 'Nina Gold', 'Jina Jay', 'Frank Marshall', 'Bonnie Timmermann', 'Ethan Coen', 'George Lucas', 'Kathleen Kennedy', 'Woody Allen', 'Michael Kahn', 'Neal H. Moritz', 'Nancy Nayor', 'Wes Craven', 'Christopher Assells', 'Jeanne McCarthy', 'Ron Howard', 'Robert Zemeckis', 'Glenn Freemantle', 'David Newman', 'Ivan Reitman', 'Christopher Boyes', 'Roger Birnbaum', 'Gary A. Hecker', 'David Rubin', 'Randi Hiller', 'Ryan Kavanaugh', 'Oliver Stone', 'Michael Bay', 'Jerry Bruckheimer', 'Mali Finn', 'Colleen Atwood', 'Harry Gregson-Williams', 'Alexandre Desplat', 'Francis Ford Coppola', 'Mike Fenton', 'Michael De Luca', 'Chris Lebenzon', 'Dean Semler', 'Roger Deakins', 'Amanda Mackey', 'Michael Kamen', 'Toby Emmerich', 'Gary Barber', 'Stephen King', 'Lisa Beach', 'Brian Tyler', 'Dottie Starling', 'Ellen Chenoweth', 'Cathy Sandrich', 'John Carpenter', 'John Davis', 'Theodore Shapiro', 'Lucinda Syson', 'Tim Burton', 'Mark Gordon', 'Peter Burgis', "Kevin O'Connell", 'Chris Columbus', 'John Lasseter', 'Kevin Smith', 'David B. Nowell', 'Richard Linklater', 'Terri Taylor', 'Akiva Goldsman', 'Camille Friend', 'Margery Simkin', 'John Ottman', 'Judd Apatow', 'Michael Mann', 'Peter Jackson', 'Randall Poster', 'Thomas Tull', 'Randy Edelman', 'Paul Schnee', 'Brian N. Bentley', 'Trevor Rabin', 'Walter F. Parkes', 'Sarah Katzman', 'Michael Minkler', 'Pete Anthony', 'Jim Passon', 'John Morris', 'Skip Lievsay', 'James Cameron', 'Guillermo del Toro', 'Andy Nelson', 'Ron Bartlett', 'Quentin Tarantino', 'Victoria Thomas', 'Gary Lucchesi', 'Steve Boeddeker', 'John Roesch', 'M. Night Shyamalan', 'Lynn Stalmaster', 'Jason Blum', 'Scott Martin Gershin', 'John Marzano', 'Mark Irwin', 'Sylvester Stallone', 'Martin Scorsese', 'Paul W.S. Anderson', 'Paul Massey', 'Phil Tippett', 'Gary Burritt', 'Tom Rosenberg', 'Menahem Golan', 'Louise Frogley', 'Barry Levinson', 'Juliet Taylor', 'Matthew F. Leonetti', 'Kevin Feige', 'Walter Hill', 'Heike Brandstatter', 'Coreen Mayrs', 'Randy Thom', 'Jennifer L. Smith', 'Gary Rydstrom', 'Mark Mothersbaugh', 'Douglas Aibel', 'Elmer Bernstein', 'John C. Stuver', 'Per Hallberg', 'Spike Lee', 'Ellen Lewis', 'Adam McKay', 'Richard King', 'Brian Avery', 'Evan Goldberg', 'Peter Farrelly', 'Barbara Harris', 'Dale E. Grahn', 'Adam Sandler', 'Tom Johnson', 'Oliver Wood', 'David Koepp', 'Sheila Jaffe', 'Frank Masi', 'Tony Scott', 'Bobby Farrelly', 'Susie Figgis', 'Rob Reiner', 'Robert Richardson', 'Harold Ramis', 'David Lynch', 'Rachel Portman', 'Mary Hidalgo', 'Juel Bestrop', 'Helen Jarvis', 'David James', 'Ellen Mirojnick', 'Howard Feuer', 'Ray Fisher', 'Dane A. Davis', 'Patrick Doyle', 'Gregg Landaker', 'Brian Helgeland', 'Gale Anne Hurd', 'Anne McCarthy', 'Richard Donner', 'Michael Semanick', 'Don Carmody', 'Sydney Pollack', 'Lawrence Gordon', 'Blake Edwards', 'Michael Shamberg', 'Pete Romano', 'Michael Hertlein', 'Ed Shearmur', 'Merrick Morton', 'Peter Hyams', 'Philippe Rousselot', 'Dino Dimuro', 'Mychael Danna', 'James Bamford', 'Wes Anderson', 'Jack Giarraputo', 'Marko A. Costanzo', 'Allen Hall', 'Justin Hammond', 'Karin Silvestri', 'Simon Kinberg', 'Christopher Nolan', 'Don Burgess', 'Pam Dixon', 'Roland Emmerich', 'Michael Giacchino', 'Robert Elswit', 'Hugo Dominguez', 'Dean Cundey', 'Roy Lee', 'David S. Goyer', 'Gus Van Sant', 'Conrad Buff IV', 'Sam Raimi', 'Jan de Bont', 'Rosemary Brandenburg', 'John Hubbard', 'Robert Troy', 'Yoram Globus', 'Richard L. Anderson', 'James Baker', 'David V. Butler', 'Doug Hemphill', 'Mary Zophres', 'Roger Mussenden', 'Pietro Scalia', 'George Miller', 'Mark Edward Wright', 'Renny Harlin', 'Lindsay Graham', 'Marc Shaiman', 'Terry Porter', 'Cameron Crowe', 'Edward Zwick', 'Michael Keller', 'John T. Reitz', 'Jane Feinberg', 'Frank Miller', 'Priscilla John', 'David Zucker', 'Mark Goldblatt', 'Joseph Middleton', 'Seth Rogen', 'Shawn Levy', 'Pedro Almod√≥var', 'Edouard F. Henriques', 'Terry Rossio', 'Nathan Kahane', 'Jeffrey Wilhoit', 'Stacey Sher', 'Gregory Nicotero', 'Wylie Stateman', 'Stuart Baird', 'Dylan Goss', 'Clayton Townsend', 'Jon Jashni', 'Phil Pastuhov', 'Johanna Ray', 'Frederick H. Stahly', 'Mark Johnson', 'Steven Rosenblum', 'Duane Manwiller', 'Todd Phillips', 'Melinda Sue Gordon', 'Lauren Shuler Donner', 'Julio Macat', 'Andrew Stanton', 'David C. Robinson', 'Klaus Badelt', 'Barry Sonnenfeld', 'Stephen Hunter Flick', 'Joel Cox', 'Steve Golin', 'Jon Title', 'Marion Dougherty', 'Dennis McNeill', 'Tessa Ross', 'Ronald Bass', 'Jackie Burch', 'George A. Romero', 'Lilly Wachowski', 'Lana Wachowski', 'Craig Henighan', 'Deirdre Bowen', 'Christian Wagner', 'Jean Ann Black', 'Michael Tronick', 'Marlene Stewart', 'Peter Staubli', 'Michael Paseornek', 'Don Zimmerman', 'John Bruno', 'Michael Ballhaus', 'Andrew Cooper', 'Joel Schumacher', 'Charles Roven', 'George Clooney', 'Peter Deming', 'Sean Daniel', 'Harry Cohen', 'Frank A. Monta√±o', 'Laura Rosenthal', 'Lon Bender', 'Dante Spinotti', 'Lora Hirschberg', 'Deborah La Mia Denaver', 'Gregg Barbanell', 'Tyler Perry', 'Avi Lerner', 'Dave Jordan', 'Alex Rouse', 'Lewis Goldstein', 'Chris Haarhoff', 'Don Bluth', 'William Goldenberg', 'Greg P. Russell', 'Steven Zaillian', 'Maher Ahmad', 'Nancy Haigh', 'Ben Waisbren', 'Anna Behlmer', 'Richard Vane', 'Dariusz Wolski', 'Mark Bridges', 'Aaron Zigman', 'Matthew Vaughn', 'Robert Greenhut', 'Michael Andrews', 'Ted Field', 'Ted Elliott', 'Joe Roth', 'Janusz Kami≈Ñski', 'Kristi Zea', 'Billy Weber', 'Kimaree Long', 'Bill Abbott', 'Stephen Goldblatt', 'Jeremy Braben', 'Alicia Stevenson', 'Colin Anderson', 'Laura Harris Atkinson', 'Karen Baker Landers', 'Teresa Eckton', 'Victoria Burrows', 'Joe Medjuck', 'Stephen S. Campanelli', 'Basil Poledouris', 'Robin D. Cook', 'Lucy Bevan', 'J. Michael Riva', 'Laurie MacDonald', 'Marci Liroff', 'John Seale', 'David Tattersall', 'James G. Robinson', 'Richard Francis-Bruce', 'Bob Badami', 'Nerses Gezalyan', 'Robert Gould', 'Paul Hirsch', 'Nancy Meyers', 'Henry Jackman', 'Paul Weitz', "Chris O'Connell", 'Robert Simonds', 'Eli Roth', 'Gary Summers', 'Nicholas Stoller', 'Terry Gilliam', 'Brian De Palma', 'J.J. Abrams', 'Bill Bernstein', 'Sheldon Kahn', 'Deborah Lynn Scott', 'Gregory Lundsgaard', 'Robert Jackson', 'Patrick Lussier', 'Robert Mark Kamen', 'Mary Selway', 'Stefan Sonnenfeld', 'Mark Helfrich', 'Brett Ratner', 'William Hoy', 'Ennio Morricone', 'Debbie McWilliams', 'Newton Thomas Sigel', 'Sanja Milkovic Hays', 'Kathy Nelson', 'David Ayer', 'Bill Pankow', 'Daniel P. Hanley', 'Chris Jenkins', 'Jack Stew', 'Kerry Lyn McKissick', 'Andrzej Bartkowiak', 'Lora Kennedy', 'Karen Golden', 'Rick Kline', 'David Orr', 'Trevor Jones', 'Walter Murch', 'Maurice Jarre', 'Dario Marianelli', 'Liza Chasin', 'George Fenton', 'Albert Wolsky', 'Paul Rubell', 'John Bailey', 'Joseph P. Reidy', 'Dan Hegeman', 'Rolfe Kent', 'Stephen Mirrione', 'Catherine Harper', 'P. Scott Sakamoto', 'Kevin Stitt', 'Alex Kurtzman', 'Adam Milo Smalley', 'Lorenzo di Bonaventura', 'Mark A. Mangini', 'Mel Metcalfe', 'Venus Kanani', 'Leslie A. Pope', 'Scott Sanders', 'Gregg Rudloff', 'Fran√ßois Duhamel', 'Ruth E. Carter', 'Hugo Weng', 'Joanna Johnston', 'Bill Conti', 'Gene Serdena', 'Shay Cunliffe', 'Denise Di Novi', 'Sidney Kimmel', 'Gail Stevens', 'Avi Arad', 'Tim Gomillion', 'Ben Stiller', 'Tyler Bates', 'John Barry', 'Fiona Weir', 'Shannon Mills', 'David C. Hughes', 'Noah Baumbach', 'Roman Polanski', 'Robert Leighton', 'Bob Beemer', 'Danny DeVito', 'Jeffrey L. Kimball', 'Gary Goldman', 'Steve Jablonsky', 'James Muro', 'Ann Roth', 'Marc Fishman', 'Wolfgang Petersen', 'Phedon Papamichael', 'David Luckenbach', 'Jann Engel', 'Anne V. Coates', 'Larry McConkey', 'Tom Stern', 'Mike Hill', 'David Appleby', 'Andrew Dunn']
        for temp_crew in df['crew']:
            myvalues_crew = [i['name'] for i in temp_crew if 'name' in i]
            pop_crew=list(set(lst2_crew).intersection(set(myvalues_crew)))
            df['crew_score']=len(pop_crew)
        '''features = ['budget', 
                'popularity', 
                'runtime', 
                'nb_spoken_languages', 
                'nb_production_companies',
                'english_spoken', 
                'language',
                'Day',
                'quarter', 'Year', 
                'Month2', 'Month3',  'Month4', 'Month5',  'Month6', 'Month7',
                'Ratiobudgetbypopularity', 'RatiopopularitybyYear',
                'RatoioruntimebyYear', 'budget_runtime_ratio', 'budget_Year_ratio',
                'Month8', 'Month9',  'Month10', 'Month11', 'Month12','dayofweek1','dayofweek2','dayofweek3','dayofweek4','dayofweek5','dayofweek6','cast_score','crew_score']
        my_genres = [g for g in genres if g != 'TV Movie']
        #features += [col for col in df.columns if 'dayofweek' in col and col != "dayofweek"]
        features += my_genres[:-1]
        features += ['production_companies_reduced_{}'.format(i) for i in range(n_comp_production_companies)]
        features += ['production_countries_reduced_{}'.format(i) for i in range(n_comp_production_countries)]'''
        features=['budget', 'popularity', 'runtime', 'nb_spoken_languages', 'nb_production_companies', 
        'english_spoken', 'language', 'Day', 'quarter', 'Year', 'Month2', 'Month3', 'Month4', 'Month5', 'Month6', 
        'Month7', 'Ratiobudgetbypopularity', 'RatiopopularitybyYear', 'RatoioruntimebyYear', 'budget_runtime_ratio', 
        'budget_Year_ratio', 'Month8', 'Month9', 'Month10', 'Month11', 'Month12', 'cast_score', 'crew_score', 
        'dayofweek1', 'dayofweek2', 'dayofweek3', 'dayofweek4', 'dayofweek5', 'dayofweek6', 'Comedy', 'Drama', 
        'Family', 'Romance', 'Horror', 'Thriller', 'Documentary', 'Action', 'Music', 'Adventure', 'Crime', 
        'Science Fiction', 'Mystery', 'Foreign', 'Animation', 'Fantasy', 'War', 'Western', 'production_companies_reduced_0', 
        'production_companies_reduced_1', 'production_companies_reduced_2', 'production_countries_reduced_0', 'production_countries_reduced_1', 'production_countries_reduced_2']

        df['language'] = df.original_language.apply(lambda x: dict_language[x])
        #df=df.drop(['dayofweek7'],axis=1)
        X = df[features]
        #X=X.drop(['dayofweek7'],axis=1)
        #df=X
        return X
        


    # x_test=pd.read_csv('./data/fwbo_kaggle_corrected.csv')
    # model_1 = load_model('trained_models/neural_network_model_1.h5')

    def __call__(self, input_vec):
        x_test=self.pre_processing(input_vec)
        pred_val_model_1=self.model_1.predict(x_test)
        #print(pred_val_model_1)
        #print(np.expm1(pred_val_model_1))
        pred_val_model_2=self.model_2.predict(x_test)
        #print(np.expm1(pred_val_model_2))
        pred_val_model_3=self.model_3.predict(x_test)
        #print(np.expm1(pred_val_model_3))
        test_data_ensemble = pd.DataFrame({'result1':pred_val_model_1.reshape(pred_val_model_1.shape[0]),
                                         'result2':pred_val_model_2.reshape(pred_val_model_2.shape[0]),
                                         'result3':pred_val_model_3.reshape(pred_val_model_3.shape[0]),
                                         })
        #pred_val_model_4=self.model_4.predict(test_data_ensemble)
        pred_val_model_1=np.expm1(pred_val_model_1)
        pred_val_model_2=np.expm1(pred_val_model_2)
        pred_val_model_3=np.expm1(pred_val_model_3)
        final_predicted_value=(pred_val_model_1+pred_val_model_2+pred_val_model_3)/3.0
 
        #final_predicted_value=np.expm1((pred_val_model_1+pred_val_model_2+pred_val_model_3/3.0))
        return final_predicted_value
        

   
