import sys
import os
import queue

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))  # noqa
from models.metadata.MDPred1 import MDPred1
from models.metadata.MDPred2 import MDPred2
from models.metadata.MDPred3 import MDPred3

from models.metadata.crawl_by_id import Crawler
from models.youtube.face_evoLVe_PyTorch.align.YTPredictor import YTPredictor


class ModelHandler:
    def __init__(self):
        self.crawler = Crawler()
        self.predictor1 = MDPred1()
        self.predictor2 = MDPred2()
        self.predictor3 = MDPred3()

        current_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        abs_path_to_youtube = os.path.abspath(os.path.join(current_dir, os.pardir, 'models', 'youtube'))
        print(abs_path_to_youtube)
        self.ytPredictor = YTPredictor(absPath_to_youtube=abs_path_to_youtube, cast_length=10, compress_width=400,
                                       skip_frames=15, frame_range=None)
        self.FIFO = queue.Queue()
        self.counter = 0
        self.solved = dict()

    def submit(self, entry_type, entry):
        self.counter += 1
        token = str(self.counter)
        self.FIFO.put((token, entry_type, entry))
        self.solved[token] = dict()
        self.solved[token]['done'] = False
        self.solved[token]['results'] = dict()
        return token

    def ask(self, token):
        if token in self.solved:
            return self.solved.get(token)
        else:
            return None

    def __call__(self, testing=None):

        while True:
            print('ModelHandler looking for next in queue')
            temp = self.FIFO.get(block=True)
            token = temp[0]
            entry_type = temp[1]  # type can be 'IMDB' 'YT' or 'BOTH
            entry = temp[2]
            print('ModelHandler processing ', token, entry_type, entry)

            if entry_type == 'IMDB':
                imdb_id = entry.get('IMDB')
                print(imdb_id)
                df = self.crawler.crawl_by_id(movie_id=imdb_id)
                df_dict = self.process_crawler(df)
                self.solved[token]['metadata'] = df_dict

                try:
                    diego = self.predictor1(df)[0]
                    self.solved[token]['results']['diego'] = int(diego)
                except:
                    print('MD Model 1 failed')
                    pass

                try:
                    venkatesh = self.predictor2(df)[0]
                    self.solved[token]['results']['venkatesh'] = int(venkatesh)
                except:
                    print('MD Model 2 failed')
                    pass

                try:
                    nikos = self.predictor3(['nm'+member.get('id') for member in df.cast[0]])
                    self.solved[token]['results']['nikos'] = int(nikos)
                except:
                    print('MD Model 3 failed')
                    pass

                self.solved[token]['done'] = True
                pass

            elif entry_type == 'YT':
                youtube_url = entry.get('YT')
                print(youtube_url)
                keep_cast_list = None
                iteration = 0
                for cast_list in self.ytPredictor.yield_faces(yt_url=youtube_url):
                    iteration += 1
                    keep_cast_list = cast_list
                    max_value = 0
                    for cast in cast_list:
                        c = cast[1][0]
                        if c > max_value:
                            max_value = c
                    self.solved[token]['youtube'] = [
                        {"value": cast[1][0] / max_value * 100, "artist": " ".join(str(cast[1][1]).split("_"))} for cast
                        in cast_list]

                    if iteration%15 == 0:
                        ids = ['nm' + cast[0] for cast in keep_cast_list]
                        relative_freq = [cast[1][0] for cast in keep_cast_list]

                        final_result = self.predictor3(ids, relative_freq)
                        self.solved[token]['results']['video'] = int(final_result)

                    # for cast in cast_list:
                    # print("imdb id:", cast[0])
                    # print("predicted label:", cast[1][1])
                    # print("the accumulated show-up-frequency:", cast[1][0])
                    # print("rank-of-popularity:", cast[1][2])
                    # self.solved[token]['youtube'][cast[1][1]] = cast[1][0]
                    print(self.solved[token]['youtube'])

                ids = ['nm'+cast[0] for cast in keep_cast_list]
                relative_freq = [cast[1][0] for cast in keep_cast_list]

                final_result = self.predictor3(ids, relative_freq)
                self.solved[token]['results']['video'] = int(final_result)
                self.solved[token]['done'] = True
                pass

            elif entry_type == 'BOTH':
                imdb_id = entry.get('IMDB')
                print(imdb_id)
                df = self.crawler.crawl_by_id(movie_id=imdb_id)
                df_dict = self.process_crawler(df)
                self.solved[token]['metadata'] = df_dict

                try:
                    diego = self.predictor1(df)[0]
                    self.solved[token]['results']['diego'] = int(diego)
                except:
                    print('MD Model 1 failed')
                    pass

                try:
                    venkatesh = self.predictor2(df)[0]
                    self.solved[token]['results']['venkatesh'] = int(venkatesh)
                except:
                    print('MD Model 2 failed')
                    pass

                try:
                    nikos = self.predictor3(['nm' + member.get('id') for member in df.cast[0]])
                    self.solved[token]['results']['nikos'] = int(nikos)
                except:
                    print('MD Model 3 failed')
                    pass

                youtube_url = entry.get('YT')
                print(youtube_url)
                keep_cast_list = None
                iteration = 0
                for cast_list in self.ytPredictor.yield_faces(yt_url=youtube_url):
                    iteration += 1
                    keep_cast_list = cast_list
                    max_value = 0
                    for cast in cast_list:
                        c = cast[1][0]
                        if c > max_value:
                            max_value = c
                    self.solved[token]['youtube'] = [
                        {"value": cast[1][0] / max_value * 100, "artist": " ".join(str(cast[1][1]).split("_"))} for cast
                        in cast_list]

                    if iteration % 15 == 0:
                        ids = ['nm' + cast[0] for cast in keep_cast_list]
                        relative_freq = [cast[1][0] for cast in keep_cast_list]

                        final_result = self.predictor3(ids, relative_freq)
                        self.solved[token]['results']['video'] = int(final_result)

                    # for cast in cast_list:
                    # print("imdb id:", cast[0])
                    # print("predicted label:", cast[1][1])
                    # print("the accumulated show-up-frequency:", cast[1][0])
                    # print("rank-of-popularity:", cast[1][2])
                    # self.solved[token]['youtube'][cast[1][1]] = cast[1][0]
                    print(self.solved[token]['youtube'])

                ids = ['nm' + cast[0] for cast in keep_cast_list]
                relative_freq = [cast[1][0] for cast in keep_cast_list]

                final_result = self.predictor3(ids, relative_freq)
                self.solved[token]['results']['video'] = int(final_result)
                self.solved[token]['done'] = True


                pass

            if testing is not None:
                break

    def process_crawler(self, df):
        title = df.original_title[0]
        genres = df.genres[0]
        cast = [member.get('name') for member in df.cast[0]]
        release_date = df.release_date[0]
        runtime = df.runtime[0]
        diction = {
            'title': title,
            'genres': genres,
            'cast': cast[0:14],
            'release_date': release_date,
            'runtime': runtime
        }
        return diction
