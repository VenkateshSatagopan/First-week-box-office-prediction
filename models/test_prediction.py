from metadata.crawl_by_id import *
from metadata.MDPred1 import *
import shutil
shutil.rmtree('metadata/webdriver/temp/')
a=Crawler()
df=a.crawl_by_id(movie_id='tt6105098')
a.close()
m1=MDPred1()
pred = m1(df)
print(pred)
