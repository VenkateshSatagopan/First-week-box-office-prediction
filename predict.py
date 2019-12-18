import pandas as pd
import shutil
# import sys
import warnings
#import numpy as np
from models.metadata.MDPredictor import MDPredictor
from models.metadata.crawl_by_id import Crawler
from models.metadata.MDPred3 import MDPred3
from models.metadata.MDPred_v import MDPred_v

# Hide warnings
warnings.filterwarnings("ignore")
# Reading the csv file
df_test = pd.read_csv("test.csv") # str(sys.argv[1])


# setting up for the prediction
shutil.rmtree('models/metadata/webdriver/temp/')
a = Crawler()
M = MDPredictor()
model3 = MDPred3()
mv = MDPred_v()

# print(df_test)
url_list = df_test.url.to_list()
id_list = df_test.imdb_id.to_list()
# print(id_list)
pred_list_1 = []
pred_list_2 = []
pred_list_3 = []
pred_list_4 = []
for i in range(len(id_list)):
    df = a.crawl_by_id(movie_id=id_list[i])
    result = M(df)
    print(result)
    # pred = (result[0]+result[1])/2
    # print('The result of movie '+str(i)+' is '+str(pred))
    pred_list_1.append(result[0][0])
    print(result[0][0])
    pred_list_2.append(result[1][0])
    print(result[1][0])
    list_of_actor_ids = ['nm'+actor['id'] for actor in df.cast[0]]
    #model3_prediction = model3(list_of_actor_ids)
    #pred_list_3.append(model3_prediction)
    #print(model3_prediction)
    #mv_pred = mv.__call__(url=url_list[i])
    #print(mv_pred)
    #pred_list_4.append(mv_pred)
    print(str(i) + "/" + str(len(id_list)))
    #print(pred_list_1)
    #print(pred_list_2)
    #print(pred_list_3)
    #print(pred_list_4)
print(pred_list_1)
print(pred_list_2)
#print(pred_list_3)
#print(pred_list_4)
df_test['meta_1'] = pred_list_1
df_test['meta_2'] = pred_list_2
#df_test['meta_2'] = pred_list_2
#df_test['meta_3'] = pred_list_3
#df_test['trailer'] = pred_list_4


# Write to csv
# print(df_test)
df_test.to_csv("test-set.csv", index=False)
print("exit python")
