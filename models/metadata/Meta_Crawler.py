from bs4 import BeautifulSoup
from urllib.request import urlopen
import urllib.parse as urlparse
import re
import random
import webbrowser
import requests
from re import sub
from decimal import Decimal
import numpy as np
import pandas as pd
import urllib
from selenium import webdriver
import calendar
from imdb import IMDb

# coding: utf-8

# In[1]:


def find_box_office(movie_id=None,driver=None):
    url_base = "https://pro.imdb.com/title/"
    url_search = url_base+movie_id+"/boxoffice"
    driver.get(url_search)
    soup_search = BeautifulSoup(driver.page_source,features='lxml')
    div_tag = soup_search.find('div',{"class":"a-section a-spacing-small opening_wknd_summary"})
    if div_tag == None:
        return None
    wbox_office = div_tag.find('div',{"class":"a-column a-span5 a-text-right a-span-last"}).get_text()
    wbox_office = Decimal(re.sub(r'[^\d.]', '', wbox_office))
    return wbox_office


# In[2]:


def find_budget(movie_id=None,driver=None):
    url_base = "https://pro.imdb.com/title/"
    url_search = url_base+movie_id
    driver.get(url_search)
    soup_search = BeautifulSoup(driver.page_source,features='lxml')
    div_tag = soup_search.find('div',{"class":"a-column a-span5 a-text-right a-span-last"})
    if div_tag == None:
        return None
    budget = div_tag.get_text()
    budget = Decimal(re.sub(r'[^\d.]', '', budget))
    return budget


# In[3]:


def find_pop(movie_id=None,driver=None):
    url_base = "https://www.imdb.com/title/"
    url_search = url_base+movie_id+"/"
    driver.get(url_search)
    soup_search = BeautifulSoup(driver.page_source,features='lxml')
    span_tag = soup_search.find_all('span',{"class":"subText"})
    try:
        result = re.sub(" ","",span_tag[2].get_text())
    except:
        return None
    pop = result.split('\n')[1]
    pop = Decimal(re.sub('[^0-9]','',pop))
    return pop


# In[4]:


def get_date(date_info=None):
    month_cal = dict((v,k) for v,k in zip(calendar.month_abbr[1:], range(1, 13)))

    day = date_info[0:2]
    mon = date_info[3:6]
    mon = month_cal[mon]
    year = date_info[7:11]
    date = str(day)+'/'+str(mon)+'/'+str(year)
    return date

