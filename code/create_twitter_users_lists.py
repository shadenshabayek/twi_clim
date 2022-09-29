import ast
import csv
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
                    import_dict,
                    save_data,
                    save_dict)

import selenium
import re

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

def get_stat(df, p, variable):

    median = df[variable].median()
    percentile = np.percentile(df[variable], p)
    #print('Median {}'.format(variable), median)
    #print('cutt-off', percentile, 'of followers count, the', p, 'percentile')

def get_top_10k(df, variable):

    df = df[df[variable]>9999]
    df['username'] = df['username'].str.lower()

    return df

'''delayers'''

def get_urls_desmog_list(filename):

    df = import_data (filename)
    if 'Unnamed: 1' in df.columns:
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
    "\|": " ",
    "douglass": "douglas",
    " for ": " ",
    " of ": " ",
    " to ": " ",
    "'": "",
    "&": "",
    " a ": " ",
    " ": "-"
     }

    df['user'] = df['user'].replace(dict, regex=True)

    dict2 = import_dict("dict2")
    df['user'] = df['user'].replace(dict2, regex=True)
    df['user'] = df['user'].str.replace('donorstrust', 'who-donors-trust')
    df['user'] = df['user'].str.replace('center-tennessee','center-of-tennessee')

    df['user'] = df['user'].str.strip()

    df['user'].loc[df['user'].str.startswith('the-')] = df['user'].str.replace("the-","")
    df['user'].loc[df['user'].str.startswith('congress-')] = df['user'].str.replace("congress-", "congress-of-")
    df['user'].loc[df['user'].str.startswith('franklin-center')] = df['user'].str.replace('franklin-center', 'franklin-centre')
    df['user'].loc[df['user'].str.startswith('citizens-the')] = df['user'].str.replace('citizens-the', 'citizens-for-the')


    list_words1 = ['environment', 'advancement', 'american', 'stewardship', 'future', 'study', 'defense']
    for word in list_words:
        df['user'].loc[df['user'].str.contains('the-{}'.format(word))] = df['user'].str.replace("the-{}".format(word),"-{}".format(word))

    list_word2 = ['capital-formation', 'affordable-energy']

    for word in list_word2:
        df['user'].loc[df['user'].str.endswith('{}'.format(word))] = df['user'].str.replace("{}".format(word),"-for-{}".format(word))

    df['user'].loc[df['user'].str.endswith('-journal')] = df['user'].str.replace("-journal","")
    df['user'].loc[df['user'].str.endswith('-cfact')] = df['user'].str.replace("-cfact","")
    df['user'].loc[df['user'].str.endswith('-advisors')] = df['user'].str.replace("-advisors", "-public-relations")
    df['user'].loc[df['user'].str.endswith('sciencecom')] = df['user'].str.replace("sciencecom","science-com")
    df['user'].loc[df['user'].str.endswith('secondstreetorg')] = df['user'].str.replace("secondstreetorg","secondstreet-org")
    df['user'].loc[df['user'].str.endswith('blacks-energy')] = df['user'].str.replace("blacks-energy", 'blacks-in-energy')
    df['user'].loc[df['user'].str.contains('-in-science')] = df['user'].str.replace('-in-science', '-science')
    df['user'].loc[df['user'].str.contains('-on-')] = df['user'].str.replace("-on-","-")
    df['user'].loc[df['user'].str.contains('steamboat')] = df['user'].str.replace('steamboat','the-steamboat')
    df['user'].loc[df['user'].str.contains('-women-s-voice')] = df['user'].str.replace("-women-s-voice","-womens-voice")


    list_users = df['user'].tolist()
    list =['https://www.desmog.com/' + user for user in list_users]

    df1 = pd.DataFrame(list, columns = ['url_desmog'])

    timestr = time.strftime("%Y_%m_%d")
    name = filename[:-4]
    save_data(df1, 'url_{}_'.format(name) + timestr + '.csv', 0)

    return df1

def get_twitter_handles_desmog_climate(collection_interupted, filename):

    timestr = time.strftime("%Y_%m_%d")
    filename_output = './data/handles' + filename
    df = get_urls_desmog_list(filename)

    if collection_interupted == 0:
        list_url = df['url_desmog'].tolist()

    elif collection_interupted == 1:
        df_collected = import_data(filename_output)
        list_url_all = df['url_desmog'].tolist()
        list_url_coll = df_collected['url_desmog'].tolist()
        list_url = [x for x in list_url_all if x not in list_url_coll]


    with open(filename_output, 'w+') as csv_file:

        writer = csv.writer(csv_file, delimiter=',')
        writer.writerow(['urls_desmog', 'tw_handle'])

        for url in list_url:

            print(url)
            browser = webdriver.Chrome(ChromeDriverManager().install())
            browser.get(url)

            try:
                element = browser.find_element_by_xpath("//h2[@id='h-social-media']/following-sibling::ul")
                txt = element.text
                handle = re.findall(r"@(.*?) on Twitter", txt)
            except:
                handle = []

            browser.quit()
            print(handle)

            writer.writerow([url, handle])
            sleep(2)

def clean_handles(df, var):

    for index, row in df.iterrows():
        df.at[index, var]=ast.literal_eval(row[var])

    list_users = [item for sublist in df[var].tolist() for item in sublist]
    list_users = list(map(str.lower, list_users))
    list_users = list(map(str.strip, list_users))

    return list_users

def clean_tw_handles_delayers():

    df1 = import_data('tw_handles_climate.csv')
    df2 = import_data('openfeedback_users.csv')
    df3 = import_data('desmog_users_politicians.csv')
    df4 = import_data('handlesdesmog_climate_org.csv')

    list_desmog_all = clean_handles(df = df1, var = 'twitter_handle')
    list_desmog_org = clean_handles(df = df4, var = 'tw_handle')

    list_desmog_politicians = df3['username'].str.lower().tolist()

    list_desmog = [x for x in list_desmog_all if x not in list_desmog_politicians]
    list_openfeedback = df2['twitter_handle'].str.lower().tolist()

    #print(list_desmog)
    #print(list_openfeedback)

    return list_desmog, list_openfeedback, list_desmog_org

def get_users_followers_delayers():

    load_dotenv()
    timestr = time.strftime("%Y_%m_%d")

    list_desmog, list_openfeedback, list_desmog_org = clean_tw_handles_delayers()

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_desmog ,
                    filename = os.path.join('.', 'data', 'followers_twitter_delayers_climate_' + timestr  + '.csv'),
                    source = 'desmog_climate_database_ind')

    # get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
    #                 list = list_openfeedback ,
    #                 filename = os.path.join('.', 'data', 'followers_twitter_delayers_climate'  + '.csv'),
    #                 source = 'open_feedback')

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_desmog_org ,
                    filename = os.path.join('.', 'data', 'followers_twitter_delayers_climate_'  + timestr + '.csv'),
                    source = 'desmog_climate_database_org')

def get_list_delayers():

    timestr = '2022_07_27'
    df = pd.read_csv('./data/followers_twitter_delayers_climate_' + timestr + '.csv', dtype='str')
    df = df[~df['detail'].notna()]
    df['follower_count'] = df['follower_count'].astype('int64')
    df['following_count'] = df['following_count'].astype('int64')

    """T. heller is in Desmog but with the twitter account with one underscore, now the second one (two under scores) suspended"""

    get_stat(df, 59, 'follower_count')
    df = get_top_10k(df, 'follower_count')

    list = df['username'].tolist()

    #print(df[df['source'] == 'open_feedback']['username'].tolist())

    return list, df

'''scientists'''

def collect_members_from_twitter_list():

    load_dotenv()

    file_name = 'members_twitter_list_scientists_who_do_climate'  + '.csv'
    filename = os.path.join('.', 'data', file_name)
    list_id = 1053067173961326594
    bearer_token = os.getenv('TWITTER_TOKEN')

    get_list_members(filename, list_id, bearer_token)

def get_list_scientists():

    df = import_data('members_twitter_list_scientists_who_do_climate'  + '.csv')
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df['id'] = df['id'].astype(str)

    df_protected = df[df['protected'] == True]

    get_stat(df, 95.75, 'follower_count')
    df = get_top_10k(df, 'follower_count')
    df= df[~df['protected'].isin([True])]
    list = df['username'].tolist()

    return list, df

'''activists'''

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
            'emmanuelfaber', #he changed his discription: "ceo - climate and social business activist" +français
            'tomvanderlee', #pol
            'kanielaing',
            'undpcambodia', #inst
            'undp_pacific' #inst
            ]

    return list

def get_list_activists():
    #source: https://www.earthday.org/19-youth-climate-activists-you-should-follow-on-social-media/
    list_manual = [
    'israhirsi',
    'xiuhtezcatl',#not found in tweets cop26 artist
    'lillyspickup',
    'jamie_margolin', #not found in tweets cop26
    'nakabuyehildaf', #not found in tweets cop26
    'namugerwaleah',#not found in tweets cop26
    'anunade', #german description but tweeted during COP26
    'varshprakash',#not found in tweets cop26
    'jeromefosterii'
    ]

    df = import_data('twitter_COP26.csv')
    df = df.dropna(subset = ['user_profile_description'])

    df['user_profile_description'] = df['user_profile_description'].str.lower()
    df['username'] = df['username'].str.lower()
    get_stat(df, 81, 'followers_count')
    df = get_top_10k(df, 'followers_count')

    list1 = filter_by_profile_description(df, ['climate', 'activist'])
    list2 = filter_by_profile_description(df, ['climate', 'justice'])
    list3 = filter_by_profile_description(df, ['environmentalist'])
    list4 = filter_by_profile_description(df, ['environmental', 'activist'])
    list5 = filter_by_profile_description(df, ['environmental', 'defender'])

    A = (set(list1) | set(list2) | set(list3) | set(list4)| set(list5))
    final_list = list(A)

    list_remove = remove_users_COP_list ()

    for user in list_remove:
        final_list.remove(user)

    return final_list

def get_users_followers_activists():

    timestr = time.strftime("%Y_%m_%d")
    list = get_list_activists()
    load_dotenv()
    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list,
                    filename = os.path.join('.', 'data', 'followers_twitter_activists_climate_' + timestr  + '.csv'),
                    source = 'cop26_hashtag')

def get_lists_and_followers():

    list_scientists, df_s = get_list_scientists()
    df_s['source'] = 'twitter_list_scientists_who_do_climate'
    print('Number of scientists', len(list_scientists))

    list_delayers, df_d = get_list_delayers()
    print('Number of delayers', len(list_delayers))

    list_activists = get_list_activists()
    print('Number of activists', len(list_activists))

    timestr = '2022_07_27'
    df_a = pd.read_csv('./data/followers_twitter_activists_climate_' + timestr + '.csv', dtype='str')
    df_a = df_a[~df_a['detail'].notna()]
    df_a['follower_count'] = df_a['follower_count'].astype('int64')
    df_a['following_count'] = df_a['following_count'].astype('int64')

    df_followers = pd.concat([df_s, df_d, df_a], axis=0, ignore_index=True)
    df_followers['source'] = df_followers['source'].str.replace('desmog_climate_database_ind', 'desmog_climate_database')
    df_followers['source'] = df_followers['source'].str.replace('desmog_climate_database_org', 'desmog_climate_database')
    
    save_data(df_followers, 'climate_groups_users_metrics_2022_08_16.csv', 0)
    print(df_followers.groupby(['source'], as_index = False).size())

    return list_scientists, list_activists, list_delayers, df_followers

def get_desmog_tw_handles():

    filename_1 = 'desmog_climate.csv'
    filename_2 = 'desmog_climate_org.csv'
    get_twitter_handles_desmog_climate(collection_interupted = 0, filename = filename_1)
    get_twitter_handles_desmog_climate(collection_interupted = 0, filename = filename_2)

def main():

    #get_desmog_tw_handles()
    #get_users_followers_delayers()
    #collect_members_from_twitter_list()
    #get_users_followers_activists()
    get_lists_and_followers()

if __name__ == '__main__':

    main()
    #get_list_delayers()
