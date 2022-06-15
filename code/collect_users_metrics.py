import numpy as np
import os
import time

from datetime import date
from dotenv import load_dotenv
from time import sleep

from utils import (get_user_metrics,
                    import_data)

def get_users_list():

    df = import_data('climate_groups_followers.csv')
    df['username'] = df['username'].str.lower()
    df = df.drop_duplicates(subset=['name'], keep='first')
    print('There are', len(df['username'].dropna().unique()), 'usernames')

    list = df['username'].tolist()

    return list

def main():

    list = get_users_list()
    timestr = time.strftime("%Y_%m_%d")

    load_dotenv()
    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                list = list,
                filename = os.path.join('.', 'data', 'user_metrics_' + timestr  + '.csv'),
                source = 'climate_groups_followers')

if __name__=="__main__":

    main()
