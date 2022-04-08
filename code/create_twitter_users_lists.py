import ast
import pandas as pd
import pickle
import numpy as np
import os
import time

pd.options.mode.chained_assignment = None  # default='warn'

from dotenv import load_dotenv
from datetime import date
from time import sleep
from utils import (get_user_metrics,
                    get_list_members,
                    import_data,
                    import_google_sheet,
                    import_dict,
                    save_data,
                    save_dict)

import selenium

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager


def get_stat(df, p, variable):

    median = df[variable].median()
    percentile = np.percentile(df[variable], p)
    print('Median {}'.format(variable), median)
    print('cutt-off', percentile, 'of followers count, the', p, 'percentile')

def get_top_10k(df, variable):

    df = df[df[variable]>9999]
    df['username'] = df['username'].str.lower()

    return df

'''delayers'''

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

    timestr = time.strftime("%Y_%m_%d")
    filename = 'tw_handles_desmog_' + timestr + '.csv'
    #df = import_data('url_desmog_climate_' + timestr + '.csv')
    df = get_urls_desmog_list()

    if collection_interupted == 0:
        list_url = df['url_desmog'].tolist()

    elif collection_interupted == 1:
        df_collected = import_data(filename)
        list_url_all = df['url_desmog'].tolist()
        list_url_coll = df_collected['url_desmog'].tolist()
        list_url = [x for x in list_url_all if x not in list_url_coll]

    list_handles = []
    list_urls = []

    for url in list_url[5:8]:

        print(url)

        browser = webdriver.Chrome(ChromeDriverManager().install())
        browser.get(url)

        try:
            element = browser.find_element_by_xpath("//h2[@id='h-social-media']/following-sibling::ul")
            print(element)
            #'//*[@id="clarify-box"]'
            twitter_handles = re.findall("@(.*?) on Twitter", element.text)

        except:
            twitter_handles = []

        browser.quit()
        print(twitter_handles)

        list_urls.append(url)
        list_handles.append(twitter_handles)
        sleep(2)

    df1 = pd.DataFrame()
    df1['urls_desmog'] = list_urls
    df1['twitter_handle'] = list_handles

    if collection_interupted == 0:
        append = 0

    elif collection_interupted == 1:
        append = 1

    save_data(df1, filename, append)

    return df1

def get_twitter_handles_desmog_openfeedback():

    df1 = import_data('tw_handles_climate.csv')
    df2 = import_data('openfeedback_users.csv')
    df3 = import_data('desmog_users_politicians.csv')

    for index, row in df1.iterrows():
        df1.at[index, 'twitter_handle']=ast.literal_eval(row['twitter_handle'])

    list_desmog_all = [item for sublist in df1['twitter_handle'].tolist() for item in sublist]
    list_desmog_all = list(map(str.lower, list_desmog_all))
    list_desmog_all = list(map(str.strip, list_desmog_all))
    list_desmog_politicians = df3['username'].str.lower().tolist()

    list_desmog = [x for x in list_desmog_all if x not in list_desmog_politicians]
    list_openfeedback = df2['twitter_handle'].str.lower().tolist()

    print(list_desmog)
    print(list_openfeedback)

    return list_desmog, list_openfeedback

def get_users_followers():

    load_dotenv()
    list_desmog, list_openfeedback = get_twitter_handles_desmog_openfeedback()

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_desmog ,
                    filename = os.path.join('.', 'data', 'followers_twitter_delayers_climate'  + '.csv'),
                    source = 'desmog_climate_database')

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_openfeedback ,
                    filename = os.path.join('.', 'data', 'followers_twitter_delayers_climate'  + '.csv'),
                    source = 'open_feedback')

def get_list_delayers():

    df = import_data ('followers_twitter_delayers_climate.csv')
    df = df[~df['description'].isin(['did not find the account, deleted or suspended'])]

    """Tony heller is in Desmog but with the twitter account with one underscore, now the second one (two under scores) suspended"""

    print('Desmog:', len(df[df['source'] == 'desmog_climate_database']))
    print('open_feedback:', len(df[df['source'] == 'open_feedback']))

    get_stat(df, 62, 'follower_count')
    df = get_top_10k(df, 'follower_count')
    list = df['username'].tolist()

    print('List Delayers:', len(list), 'users')
    return list, df

'''scientists'''

def collect_members_from_twitter_list():

    load_dotenv()

    file_name = 'members_twitter_list_scientists_who_do_climate'  + '.csv'
    filename = os.path.join('.', 'data', file_name)
    list_id = 1053067173961326594
    bearer_token = os.getenv('TWITTER_TOKEN')

    get_list_members(filename, list_id, bearer_token)

def get_list_scientists_who_do_climate():

    df = import_data('members_twitter_list_scientists_who_do_climate'  + '.csv')
    print('total number of members:', len(df))
    df = df.replace(r'^\s*$', np.nan, regex=True)

    df_protected = df[df['protected'] == True]
    print(len(df_protected))

    get_stat(df, 95.75, 'follower_count')
    df = get_top_10k(df, 'follower_count')
    df= df[~df['protected'].isin([True])]
    list = df['username'].tolist()

    print('List Scientists:', len(list), 'users')

    # #old list
    #
    # df_type = import_data('type_users_climate.csv')
    # df_type['type'] = df_type['type'].astype(int)
    #
    # keep_type = [2]
    # df_type = df_type[df_type['type'].isin(keep_type)]
    #
    # list_old = df_type['username'].tolist()
    # list_diff = [x for x in list if x not in list_old]
    # print(list_diff)

    return list, df

'''sctivists'''
def filter_by_profile_description(df, list_keywords):

    df1 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords))]
    list = df1['username'].unique().tolist()
    return list

def remove_users_COP_list ():

    list = ['bjornlomborg', #already in the dataset of desmog
            'kimjungil1984', #seems fake
            'dkfordicky', #personal pictures
            'tonylavs', #other language
            'rkyte365', #conseil
            'lisabloom', #lawyer -  mais suivi par extinction rebellion...
            'charlotte_cymru', #not always climate focused
            'jeanmanes', #embassador
            'ecosensenow', #already in the dataset of desmog
            'geoffreysupran', # among scientists who do climate
            'janet_rice', #pol
            'emmanuelfaber' #he changed his discription: "ceo - climate and social business activist" +français
            ]

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
    get_stat(df, 81, 'followers_count')
    df = get_top_10k(df, 'followers_count')

    list1 = filter_by_profile_description(df, ['climate', 'activist'])
    list2 = filter_by_profile_description(df, ['environmentalist'])
    list3 = filter_by_profile_description(df, ['environmental activist'])

    A = (set(list_manual) | set(list1) | set(list2) | set(list3))
    final_list = list(A)

    list_remove = remove_users_COP_list ()

    for user in list_remove:
        final_list.remove(user)

    print('cliamte activists:', len(final_list))
    return final_list

def main():

    #get_twitter_handles_desmog_climate(collection_interupted = 0)
    #collect_members_from_twitter_list()
    #get_twitter_handles_desmog_openfeedback()
    #get_users_followers()
    #get_list_delayers()
    #get_list_scientists_who_do_climate()
    get_list_activists()

if __name__ == '__main__':

    main()
