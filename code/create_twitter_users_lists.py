
import pandas as pd
import pickle
import numpy as np
import os
import time

from dotenv import load_dotenv
from datetime import date
from utils import (get_user_metrics,
                    import_data,
                    import_google_sheet,
                    import_dict,
                    save_data,
                    save_dict)

import selenium

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

""" Desmog database, get list climate + twitter handles"""

def get_urls_desmog_list():

    df = import_data ('desmog_climate.csv')
    df = df.drop(columns=['Unnamed: 1'], axis=1)
    df['user'] = df['user'].astype(str)
    df['user'] = df['user'].str.lower()

    dict = {
    "\.": "",
    "(deceased)": "",
    "sir ": "",
    "lord ": "",
    #" a ": " ",
    #" w ": " ",
    ":": " ",
    "\’": "-",
    "\,": "",
    "è": "-",
    "é": "e",
    "ö": "o",
    "ó": "o",
    "ø": "o",
    "á": "a",
    "å": "a",
    "ü": "u",
    "\(": "",
    "\)": "",
    "douglass": "douglas",
    " ": "-",
     }

    df['user'] = df['user'].replace(dict, regex=True)

    dict2 = import_dict("dict2")
    df['user'] = df['user'].replace(dict2, regex=True)
    df['user'] = df['user'].str.strip()

    list_users = df['user'].tolist()
    list =['https://www.desmog.com/' + user for user in list_users]

    df1 = pd.DataFrame(list, columns = ['url_desmog'])

    timestr = time.strftime("%Y_%m_%d")
    save_data(df1, 'url_desmog_climate_' + timestr + '.csv', 0)

    return df1

def get_twitter_handles_desmog_climate(collection_interupted):

    #df = get_urls_desmog_list()
    timestr = time.strftime("%Y_%m_%d")
    filename = 'tw_handles_desmog_' + timestr + '.csv'

    df = import_data('url_desmog_climate_' + timestr + '.csv')

    if collection_interupted == 0:
        list_url = df['url_desmog'].tolist()

    elif collection_interupted == 1:
        df_collected = import_data(filename)
        list_url_all = df['url_desmog'].tolist()
        list_url_coll = df_collected['url_desmog'].tolist()

        list_url = [x for x in list_url_all if x not in list_url_coll]

    list_handles = []
    list_urls = []

    for url in list_url:

        print(url)

        browser = webdriver.Chrome(ChromeDriverManager().install())
        browser.get(url)

        try:
            element = browser.find_element_by_xpath("//h2[@id='h-social-media']/following-sibling::ul")
            twitter_handles = re.findall("@(.*?) on Twitter", element.text)

        except:
            twitter_handles = []

        browser.quit()
        print(twitter_handles)

        list_urls.append(url)
        list_handles.append(twitter_handles)

    df1 = pd.DataFrame()
    df1['urls_desmog'] = list_urls
    df1['twitter_handle'] = list_handles

    #append = 1
    append = 0


    save_data(df1, filename, append)

    return df1

def get_twitter_handles_desmog_openfeedback():

    df1 = import_data('tw_handles_desmog.csv')
    df2 = import_data('openfeedback_users.csv')



def get_users_followers():

    load_dotenv()

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_twitter_handles_from_google_spreadsheet('tw_handles_climate') ,
                    filename = os.path.join('.', 'data', 'followers_twitter_desmog_climate'  + '.csv'),
                    source = 'desmog_climate_database')

def get_list_desmog():

    #df = import_google_sheet ('followers_twitter_desmog_climate')
    #if not linked to google spread sheet_instance
    df = import_data('followers_twitter_desmog_climate.csv')

    """drop politicians"""

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df_politcians = import_data('desmog_users_politicians.csv')
    df_politcians['username'] = df_politcians['username'].lower()
    drop_politicians = df_politcians['username'].tolist()

    df = df[~df['username'].isin(drop_politicians)]

    """ Take desmog users with 10k+ followers"""

    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    """Tony heller is in Desmog but with the twitter account with one underscore, now the second one suspended"""

    list = df['username'].tolist() + ['tony__heller']

    print("The number of Twitter (desmog) handles is", len(list))

    return list

def collect_members_from_twitter_list():

    load_dotenv()
    filename = './data/memebers_twitter_list_scientists_who_do_science'  + '.csv'
    list_id = 1053067173961326594
    bearer_token = os.getenv('TWITTER_TOKEN')

    get_list_members(filename, list_id, bearer_token)

    df = import_data(filename)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    return df

def get_list_scientists_who_do_climate():

    #df = import_google_sheet ('followers_twitter_list_scientists_who_do_science')
    #if not linked to google spread sheet_instance
    df = import_data('followers_twitter_list_scientists_who_do_science.csv')
    df = df.replace(r'^\s*$', np.nan, regex=True)

    """ Take users with 10k+ followers"""

    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    list = df['username'].tolist()
    #print(list)
    print("The number of Twitter (scientists who do science) handles is", len(list))

    return list

def get_list_open_feedback():

    #df = import_google_sheet ('followers_twitter_desmog_climate')
    df = import_data ('followers_twitter_desmog_climate.csv')
    df = df[df['source'] == 'EV']
    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    list = df['username'].tolist()
    print('List Open Feddback:', len(list), 'users')

    return list

def get_list_activists():
    #source: https://www.earthday.org/19-youth-climate-activists-you-should-follow-on-social-media/
    #keep those with 10k followers
    list_manual = [
    'israhirsi',
    'xiuhtezcatl',
    'lillyspickup',
    'jamie_margolin',
    'nakabuyehildaf',
    'namugerwaleah',
    'anunade',
    'varshprakash',
    'jeromefosterii',
    'the_ecofeminist', #100_top_liked_tweets_cop26
    'sumakhelena', #100_top_liked_tweets_cop26
    'gretathunberg' #100_top_liked_tweets_cop26
    ]

    df = import_data('twitter_COP26.csv')
    df = df.dropna(subset = ['user_profile_description'])
    df['user_profile_description'] = df['user_profile_description'].str.lower()
    df['username'] = df['username'].str.lower()
    df = df[df['followers_count']>9999]

    list_keywords =['climate', 'activist']
    df1 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords))]
    list_users_cop = df1['username'].unique().tolist()

    list_keywords2 = ['environmentalist']
    df2 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords2))]
    list_users_cop2 = df2['username'].unique().tolist()

    list_keywords3 = ['environmental activist']
    df3 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords3))]
    list_users_cop3 = df3['username'].unique().tolist()

    A = (set(list_manual) | set(list_users_cop) | set(list_users_cop2) | set(list_users_cop3))
    final_list = list(A)

    final_list.remove('bjornlomborg') #delayer, in the dataset of desmog!
    final_list.remove('kimjungil1984') #seems fake
    final_list.remove('dkfordicky') #pictures of himself
    final_list.remove('tonylavs') #other language
    final_list.remove('rkyte365') #conseil
    final_list.remove('lisabloom') #lawyer hors sujet ? mais suivi par extinction rebellion...
    final_list.remove('charlotte_cymru') #not always climate focused
    final_list.remove('jeanmanes') #embassador
    final_list.remove('ecosensenow') # in the dataset of desmog!
    final_list.remove('geoffreysupran') # among scientists who do climate
    final_list.remove('janet_rice') #pol
    final_list.remove('emmanuelfaber') #he changed his discription: "ceo - climate and social business activist" +français

    print('cliamte activists:', len(final_list))

    return final_list

if __name__ == '__main__':

    get_urls_desmog_list()
    #get_twitter_handles_desmog_climate(collection_interupted = 0)
    #get_list_scientists_who_do_climate()
    #get_list_open_feedback()
    #et_list_desmog()

    #get_list_top_mentioned_by_type()

    #get_users_followers()
    #get_list_activists()
    #get_twitter_handles_desmog_climate()
