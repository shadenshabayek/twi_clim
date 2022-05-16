import os
import time
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.0f' % x)

from datetime import date
from dotenv import load_dotenv
from time import sleep

from utils import (collect_liked_tweets_data,
                    tic,
                    toc)

def clean_dataframe (df):

    description = ['did not find the account, deleted or suspended']
    df = df[~df['description'].isin(description)]
    df = df[~df['protected'].isin(['True'])]

    df['follower_count'] = df['follower_count'].astype('int64')
    df['id'] = df['id'].astype('int64')
    df['following_count'] = df['following_count'].astype('int64')

    #df = df.sort_values(by = 'follower_count', ascending = False)

    return df


def get_list_users_id (data_path):

    data_path = data_path
    #important: read data as str, otherwise conversion from float64 to int64 of
    #long twitter ID can change last digit.
    df_initial = pd.read_csv(data_path, dtype='str')

    df = clean_dataframe (df_initial)

    list_users = df['username'].tolist()
    list_users_id = df['id'].tolist()
    print('total nb of users', len(list_users))

    return list_users, list_users_id

def main():

    load_dotenv()

    timestr = time.strftime("%Y_%m_%d")
    path = './data/user_metrics_2022_04_23.csv'

    list_users, list_users_id = get_list_users_id(data_path = path)
    l = len(list_users_id)

    tic()
    for i in range(0,l):
      print(i)
      collect_liked_tweets_data(list_individuals = list_users ,
                             author_id = list_users_id[i],
                             author_name = list_users[i],
                             bearer_token = os.getenv('TWITTER_TOKEN'),
                             filename = os.path.join('.', 'data', 'climate_liked_tweets_test_' + timestr + '.csv'))

      sleep(20)
    toc()

if __name__=="__main__":

    main()
