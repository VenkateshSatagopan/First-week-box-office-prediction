import os
import sys
# import calendar
import time
import pandas as pd
# from re import sub
# from decimal import Decimal
from pyvirtualdisplay import Display
from selenium import webdriver
from imdb import IMDb
from zipfile import ZipFile

from .Meta_Crawler import find_box_office
from .Meta_Crawler import find_budget
from .Meta_Crawler import find_pop
from .Meta_Crawler import get_date


class Crawler:
    def __init__(self):
        # Set the virtual display
        display = Display(visible=0, size=(800, 800))
        # unzip the linux chrome driver
        driver_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), './webdriver/'))
        driver_file = os.path.abspath(os.path.join(driver_folder, 'chromedriver_linux64.zip'))
        driver_temp_extraction_folder = os.path.abspath(os.path.join(driver_folder, './temp/'))
        driver_temp_extracted_folder = os.path.abspath(os.path.join(driver_temp_extraction_folder, 'chromedriver'))

        with ZipFile(driver_file, 'r') as zipObj:
            zipObj.extractall(driver_temp_extraction_folder)

        # allow execution
        os.chmod(driver_temp_extracted_folder, 755)

        # log in to the imdb pro
        account = "huyichen1995@163.com"
        password = "amlproject2019"
        # driver_option = webdriver.ChromeOptions()
        # driver_option.add_argument("--headless")
        display.start()
        driver = webdriver.Chrome(driver_temp_extracted_folder)  # need to set the path of the driver
        driver.get(
            "https://secure.imdb.com/ap/signin?openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.imdb.com%2Fap-signin-handler&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.assoc_handle=imdb_pro_us&openid.mode=checkid_setup&siteState=eyJvcGVuaWQuYXNzb2NfaGFuZGxlIjoiaW1kYl9wcm9fdXMiLCJyZWRpcmVjdFRvIjoiaHR0cHM6Ly9wcm8uaW1kYi5jb20vIn0&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0")
        driver.find_element_by_id("ap_email").send_keys(account)
        driver.find_element_by_id("ap_password").send_keys(password)
        driver.find_element_by_id("signInSubmit").click()
        self.driver = driver
        # in case for the robot checking
        # time.sleep(10)

    def close(self, driver=None):
        driver = self.driver
        driver.close()

    def crawl_by_id(self, movie_id=None, driver=None):
        driver = self.driver
        ia = IMDb()
        try:
            x = ia.get_movie(movie_id[2:])
        except:
            return

        # find popularity
        popularity = find_pop(movie_id=movie_id, driver=driver)

        try:
            box_office = find_box_office(movie_id=movie_id, driver=driver)
        except:
            box_office = None

        # find budget
        try:
            budget = find_budget(movie_id=movie_id, driver=driver)
            if budget == box_office:
                budget = None
        except:
            budget = None

        # find title
        try:
            title = x['title']  # title
        except:
            title = None

        # find cast
        try:
            cast_list = x['cast']  # cast_id
        except:
            cast_info = [None]
        else:
            cast_info = []
            for i in range(len(cast_list)):
                d = {'id': cast_list[i].getID(), 'name': cast_list[i]['name']}
                cast_info.append(d)

        # find run time
        try:
            run_time = x['runtimes']  # run time
        except:
            run_time = None

        # find country
        try:
            country = x['country codes']  # country codes
        except:
            country = None

        # find language
        try:
            language = x['language codes']  # language
        except:
            language = None

        # find production companies
        try:
            company_list = x['production companies']  # production companies
        except:
            company = None
        else:
            product_company = []
            for i in range(len(company_list)):
                d = {'id': company_list[i].getID(), 'name': company_list[i]['name']}
                product_company.append(d)

        # find genres
        try:
            genres = x['genres']
        except:
            genres = None

        # find rating
        try:
            rating = x['rating']
        except:
            rating = None

        # find votes
        try:
            votes = x['votes']
        except:
            votes = None

        # get crew members
        try:
            data_list = list(x.data.keys())
            crew_list = list()
        except:
            crew_list = None
        else:
            for i in range(len(data_list)):
                if 'ors' in data_list[i] or 'ers' in data_list[i] or 'depart' in data_list[i]:
                    crew_list.append(data_list[i])

            if 'distributors' in crew_list:
                crew_list.remove('distributors')

        crew_info = []
        for i in range(0, len(crew_list)):
            crew_temp = x[crew_list[i]]
        for n in range(len(crew_temp)):
            try:
                id_temp = crew_temp[n].getID()
                name_temp = crew_temp[n]['name']
            except:
                id_temp = None
                name_temp = None
            else:
                d = {'id': id_temp, 'name': name_temp}
            crew_info.append(d)

        # get release date
        try:
            date_info = x['original air date']
            date = get_date(date_info=date_info)
        except:
            date = "01/01/2019"
        

        # driver.close()

        dataset = {'budget': [budget],
                   'genres': [genres],
                   'imdb_id': [movie_id],
                   'spocken_languages': [language],
                   'original_title': [title],
                   'popularity': [popularity],
                   'production_companies': [product_company],
                   'production_countries': [country],
                   'release_date': [date],
                   'runtime': run_time,
                   'cast': [cast_info],
                   'crew': [crew_info], }
        # 'wb_info':[box_office]}
        df_meta = pd.DataFrame(dataset)
        # df_meta.to_csv('./get_meta/meta_info.csv', columns=df_meta.columns, index=False)
        return df_meta

    def __call__(self, _movie_id=None):
        return self.crawl_by_id(self, movie_id=_movie_id)
