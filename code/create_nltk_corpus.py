import pandas as pd
import numpy as np
import time
import re
import string

from create_twitter_users_lists import get_lists_and_followers
from datetime import date
from utils import (import_data,
                   save_text_file)

def add_type(var1, var2, df):

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()
    df[var1] = ''
    df[var1] = np.where(df[var2].isin(list_scientists), 'scientist', df[var1])
    df[var1] = np.where(df[var2].isin(list_activists), 'activist', df[var1])
    df[var1] = np.where(df[var2].isin(list_delayers), 'delayer', df[var1])

    return df

def get_tweets():

    df = import_data ('twitter_data_climate_tweets_2022_03_15.csv')
    df = df[~df['query'].isin(['f'])]
    df = add_type('type', 'username', df)
    df['username'] = df['username'].str.lower()
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'] + '.'
    df = df[df['lang'] == 'en']
    print('number of tweets', len(df))

    return df

def tw_preprocessor(text):

    text = re.sub(r'http\S+', '', text) #remove links
    text = text.replace('\n', ' ') #remove line breaks
    text = re.sub(r'@\S+', '', text) #remove @
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text.lower()) #remove punctuation and lower caps

    return text

def remove_covid_tweets(df):

    df['text'] = df['text'].str.lower()
    mylist = ['covid',
              'mask',
              'fauci',
              'wuhan',
              'vax',
              'pandemic',
              'corona',
              'virus',
              'pharmas']

    df1 = df[df.text.apply(lambda tweet: any(words in tweet for words in mylist))]
    list1 = df1['id'].tolist()
    print('Number of covid tweets', len(list1))

    df = df[~df['id'].isin(list1)]
    return df

def remove_tweets (df, remove_covid, set_topic_climate):

    df['text'] = df['text'].apply(tw_preprocessor)

    if remove_covid == 1:
        df = remove_covid_tweets(df)
        print('total number of tweets all time, excluding covid tweets', len(df))

    if set_topic_climate == 1:
        list_1 = ['arctic',
                    'alarmist',
                    'antarctic',
                    'bleaching',
                    'carbon ',
                    'climate',
                    'CO2',
                    'emissions',
                    ' fire'
                    'forest',
                    'geological',
                    'greenhouse',
                    'glacier',
                    'glaciers',
                    'heatwave',
                    ' ice ',
                    'nuclear',
                    ' ocean ',
                    'oceans ',
                    'plant ',
                    'pollutant',
                    'pollution',
                    'polar',
                    'renewable',
                    'recycled',
                    'recycle',
                    'science',
                    'solar',
                    'species',
                    'warming',
                    'wildfire',
                    'wildfires',
                    'wind ',
                    'wildlife',
                    'weather']
        #list from topic 0
        list_2 = ['climate',
                    'scientists' ,
                    ' heat ',
                    'drought',
                    'environmental',
                    'nature',
                    'planet',
                    'warming ',
                    ' water ',
                    'ocean',
                    'heatwaves',
                    'emissions',
                    'adaptation',
                    'planet ' ,
                    'temperatures' ,
                    'ecosystems ',
                    'research',
                    'resilience',
                    'carbon',
                    'heatwave',
                    'fossil',
                     'fuel']

        mylist = list(set(list_1) | set(list_2))

        df = df[df.text.apply(lambda tweet: any(words in tweet for words in mylist))]
        print('there are', len(df), 'tweets that speak about climate')
        print('tweets about climate per group:', (df.groupby(['type'])['text'].count())/len(df))
        print('average number of tweets per group')

    return df

def create_corpus(remove_covid, set_topic_climate, cop26):

    #timestr = time.strftime("%Y_%m_%d")
    df = get_tweets()
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    df = df[(df['date']> date(2021, 10, 30)) & (df['date']<date(2021, 11, 30))]
    print('total number of tweets', len(df))

    if cop26 == 1:
        df = df[(df['date']> date(2021, 10, 30)) & (df['date']<date(2021, 11, 13))]
        df = df.reset_index()
        print('total number of tweets COP26', len(df))

    print('total number of users', df['username'].nunique())

    df = remove_tweets (df, remove_covid, set_topic_climate)

    for user in df['username'].unique().tolist():

        df1 = df[df['username'] == user]
        type = df1['type'].iloc[0]
        text = df1['text'].tolist()

        if type == 'scientist':
            file_name = 'twi_clim_corpus/pos/' + user + '.txt'
            save_text_file(text, file_name)

        elif type == 'delayer':
            file_name = 'twi_clim_corpus/neg/' + user + '.txt'
            save_text_file(text, file_name)

        elif type == 'activist':
            file_name = 'twi_clim_corpus/neu/' + user + '.txt'
            save_text_file(text, file_name)


def main():

    create_corpus(remove_covid  = 1, set_topic_climate = 1, cop26 = 0)

if __name__ == '__main__':
    main()
