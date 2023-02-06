import os
import numpy as np
import time

from datetime import date
from dotenv import load_dotenv
from time import sleep

from utils import (collect_twitter_data,
                    import_google_sheet,
                    import_data,
                    tic,
                    toc)

from create_twitter_users_lists import get_lists_and_followers

def get_users(collection_interupted):

    df = import_data('climate_groups_followers_2022_07_28.csv')
    df = df.sort_values(by = 'follower_count', ascending = False)
    df['protected'] = df['protected'].astype(str)
    df = df[~df['protected'].isin(['True'])]
    print('number of users, after removing protected accounts', len(df))
    list_users_all = df['username'].tolist()

    if collection_interupted == 0:

        list_users = df['username'].tolist()
        print('total nb of users', len(list_users))

    elif collection_interupted == 1 :

        timestr = time.strftime("%Y_%m_%d")
        #timestr = '2022_07_19'
        df_collected = import_data('twitter_data_climate_tweets_' + timestr  + '.csv')

        list1 = df_collected.username.unique().tolist()
        print('Number of users for which the tweets were collected', len(list1))

        list2 = df['username'].dropna().unique()
        list_users = [x for x in list2 if x not in list1]

    return list_users_all, list_users

def main():

    load_dotenv()
    timestr = time.strftime("%Y_%m_%d")
    #timestr ="2022_07_19"
    list_users_all, list_users = get_users(collection_interupted = 0)
    print(len(list_users))

    list_users_tw =['from:' + user for user in list_users]
    print(len(list_users_tw))

    tic()
    for query in list_users_tw:
        collect_twitter_data(
            list_individuals = list_users_all,
            query = query,
            start_time = '2021-01-01T23:00:00Z',
            end_time = '2023-02-01T23:00:05Z',
            bearer_token= os.getenv('TWITTER_TOKEN'),
            filename = os.path.join('.', 'data', 'twitter_data_climate_tweets_' + timestr  + '.csv'),
            )
        sleep(3)
    toc()

if __name__=="__main__":

    main()
