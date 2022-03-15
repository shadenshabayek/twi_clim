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

def get_list_users(collection_interupted):

    timestr = time.strftime("%Y_%m_%d")
    df = import_data('type_users_climate.csv')
    keep_type = [1, 2, 4]
    df = df[df['type'].isin(keep_type)]
    df['username'] = df['username'].str.lower()
    list_users_all = df['username'].tolist()

    if collection_interupted == 0:
        list_users = df['username'].tolist()
    #list_users = [x for x in df['username'].unique() if x not in df1['username'].unique()]
    #print('Total number of tweets (type 1,2,4) before update:', len(df_final))
    elif collection_interupted == 1:

        df_tweets = import_data('twitter_data_climate_' + timestr  + '.csv')
        list_users = [x for x in df['username'].unique() if x not in df_tweets['username'].unique()]


    return list_users_all, list_users

if __name__=="__main__":

    load_dotenv()


    list_users_all, list_users = get_list_users(collection_interupted = 0)
    print(len(list_users))


    #list_users_tw =['from:' + user for user in list_users]
    list = ['milneorchid']
    list_users_tw =['from:' + user for user in list]

    tic()
    timestr = time.strftime("%Y_%m_%d")
    df_tweets = import_data('twitter_data_climate_' + timestr  + '.csv')
    l = len(df_tweets)
    #print(df_tweets['created_at'].iloc[l-1])
    #print(list_users_tw[11:100])

    for query in list_users_tw :

        collect_twitter_data(
            list_individuals = list_users_all,
            query = query,
            #start_time = '2021-06-01T23:00:00Z',
            start_time = '2021-06-03T01:55:20.000Z',
            end_time = '2021-06-03T09:55:20.000Z',
            #end_time = '2021-12-01T23:00:05Z',
            #end_time = df['created_at'].iloc[l-1],
            bearer_token= os.getenv('TWITTER_TOKEN'),
            #filename = os.path.join('.', 'data', 'twitter_data_climate_' + timestr  + '.csv'),
            filename = os.path.join('.', 'data', 'twitter_data_climate_test_' + timestr  + '.csv')
            )
        sleep(3)
    toc()
