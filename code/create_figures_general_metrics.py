import os
from datetime import datetime, timedelta, date
from utils import (import_data, save_figure)
from create_twitter_users_lists import get_lists_and_followers

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #new addition to create the boxes in the legend for the shaded areas
import time
import ural

def get_tweets_by_type():

    df  = import_data('twitter_data_climate_tweets_2022_07_19.csv')
    df['username'] = df['username'].str.lower()
    list_manual = [
    'israhirsi',
    'xiuhtezcatl',#not found in tweets cop26 artist
    'lillyspickup',
    'jamie_margolin', #not found in tweets cop26
    'nakabuyehildaf', #not found in tweets cop26
    'namugerwaleah',#not found in tweets cop26
    'anunade', #german description but tweeted during COP26
    'varshprakash',#not found in tweets cop26
    'jeromefosterii']

    list_2 = ['johnredwood' 'climaterealists' 'tan123' 'netzerowatch' 'electroversenet',
    'marcelcrok' 'alexnewman_jou']

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()

    df['type'] = ''
    df['type'] = np.where(df['username'].isin(list_scientists), 'scientist', df['type'])
    df['type'] = np.where(df['username'].isin(list_activists), 'activist', df['type'])
    df['type'] = np.where(df['username'].isin(list_delayers), 'delayer', df['type'])

    df = df[df['type'].isin(['scientist', 'activist','delayer'])]
    df = df[~df['username'].isin(['dailycaller'])]

    return df

def plot_format(ax, plt):

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.locator_params(axis='y', nbins=4)
    ax.xaxis.set_tick_params(length=0)

    ax.grid(axis='y')
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles=handles, loc = 'upper left')
    plt.tight_layout()

def get_df_by_group(type, variable):

    df = get_tweets_by_type()
    df = df[df['type'] == type ]

    df['type_of_tweet'] = df['type_of_tweet'].replace(np.nan, 'raw_tweet')
    df['total_engagement'] = (df['retweet_count'] + df['like_count'] + df['reply_count'])
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    df1 = df[df['type_of_tweet'] == variable].groupby(['date', 'username'], as_index=False).size()
    df_mean = df1.groupby(['date'], as_index=False)['size'].mean()

    return df_mean

def create_twitter_figure_per_user(figure_name, title, variable, y_max, list):

    df_d = get_df_by_group(type = 'delayer', variable = variable)
    df_s = get_df_by_group(type = 'scientist', variable = variable)
    df_a = get_df_by_group(type = 'activist', variable = variable)
    #df_volume_rt = df_rt.groupby(['date'], as_index=False).size()

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(df_d['date'],
        df_d['size'],
        color='salmon',
        label='Delayers')

    ax.plot(df_a['date'],
        df_a['size'],
        color='orange',
        label='Activists')

    ax.plot(df_s['date'],
        df_s['size'],
        color='lightgreen',
        label='Scientists')

    ax.set_xlim([date(2021, 7, 1), date(2021, 11, 29)])

    df_fires = import_data('wildfires_2021.csv')
    df_fires['fire_date'] = pd.to_datetime(df_fires['month']).dt.date

    for fire_date in df_fires['fire_date'].tolist():
        plt.axvline(np.datetime64(str(fire_date)), color='gray', linestyle='--', alpha=0.1)
    ax.set(
       title = title )
    ax.set_ylim(-0.1, y_max)
    ax.set_yticks(list)

    plt.axvspan(np.datetime64('2021-10-31'),
                np.datetime64('2021-11-12'),
                ymin=0, ymax=5000,
                facecolor='g',
                alpha=0.1)

    plt.xticks(rotation = 30, fontsize = 9, ha = 'right')
    plt.text(np.datetime64("2021-11-02"), 1, "COP26", fontsize=7, color='g')

    plot_format(ax, plt)

    save_figure(figure_name)

def create_twitter_figure(figure_name, title, type):

    df = get_tweets_by_type()
    df = df[df['type'] == type ]
    df['type_of_tweet'] = df['type_of_tweet'].fillna('raw_tweet')
    #replace(np.nan, 'raw_tweet')
    #df['total_engagement'] = (df['retweet_count'] + df['like_count'] + df['reply_count'])
    print(df['created_at'].iloc[0])

    df['date'] = pd.to_datetime(df['created_at']).dt.date
    print(df.groupby(['type_of_tweet'], as_index = False).size())
    df_rt = df[df['type_of_tweet'] == 'retweeted'].groupby(['date'], as_index=False).size()
    df_rp = df[df['type_of_tweet'] == 'replied_to'].groupby(['date'], as_index=False).size()
    df_qt = df[df['type_of_tweet'] == 'quoted'].groupby(['date'], as_index=False).size()
    df_cc = df[df['type_of_tweet'] == 'raw_tweet'].groupby(['date'], as_index=False).size()

    fig, ax = plt.subplots(figsize=(8, 4))

    d = df[(df['date']> date(2021, 7, 1) ) & (df['date']<date(2022, 1, 1))]
    total = d['id'].count()

    ax.plot(df_cc['date'],
        df_cc['size'],
        color='deepskyblue',
        label='Created Tweets per day')

    ax.plot(df_rt['date'],
        df_rt['size'],
        color='lightgreen',
        label='Retweets per day')

    ax.plot(df_qt['date'],
        df_qt['size'],
        color='pink',
        label='Quotes per day')

    ax.plot(df_rp['date'],
        df_rp['size'],
        color='crimson',
        label='Replies per day')

    ax.set(
       title = title )

    ax.set_xlim([date(2021, 7, 2), date(2021, 11, 29)])
    ax.set_ylim([-0.1, 2500])
    plt.axvspan(np.datetime64('2021-10-31'),
                np.datetime64('2021-11-12'),
                ymin=0, ymax=5000,
                facecolor='g',
                alpha=0.1)

    plt.axvspan(np.datetime64('2021-01-25'),
                np.datetime64('2021-06-30'),
                ymin=0,
                ymax=200000,
                facecolor='r',
                alpha=0.05)
    plt.xticks(rotation = 30, fontsize = 9, ha = 'right')
    df_fires = import_data('wildfires_2021.csv')
    df_fires['fire_date'] = pd.to_datetime(df_fires['month']).dt.date

    for fire_date in df_fires['fire_date'].tolist():
        plt.axvline(np.datetime64(str(fire_date)), color='gray', linestyle='--', alpha=0.1)

    plt.text(np.datetime64("2021-11-02"), 3, "COP26", fontsize=6, color='g')
    plot_format(ax, plt)

    save_figure(figure_name)
    #plt.show()

def create_general_activity_figures():

    timestr = time.strftime("%Y_%m_%d")

    create_twitter_figure(figure_name = 'twitter_volume_climate_scientists_'+ timestr + '.jpg',
                        title = 'Climate Scientists',
                        type = 'scientist' )

    create_twitter_figure(figure_name = 'twitter_volume_climate_activists_'+ timestr + '.jpg',
                        title = 'Climate Activists',
                        type = 'activist' )

    create_twitter_figure(figure_name = 'twitter_volume_climate_delayer_'+ timestr + '.jpg',
                        title = 'Climate Delayers',
                        type = 'delayer' )

    create_twitter_figure_per_user(figure_name = 'retweets_per_day_'+ timestr + '.jpg',
                                    title = 'Retweets per user per day',
                                    variable = 'retweeted',
                                    y_max = 35,
                                    list = [0, 5, 10, 15, 20, 25, 30])

    create_twitter_figure_per_user(figure_name = 'replies_per_day_'+ timestr + '.jpg',
                                    title = 'Replies per user per day',
                                    variable = 'replied_to',
                                    y_max = 35,
                                    list = [0, 5, 10, 15, 20, 25, 30])

    create_twitter_figure_per_user(figure_name = 'quotes_per_day_'+ timestr + '.jpg',
                                    title = 'Quotes per user per day',
                                    variable = 'quoted',
                                    y_max = 35,
                                    list = [0, 5, 10, 15, 20, 25, 30])

    create_twitter_figure_per_user(figure_name = 'cc_per_day_'+ timestr + '.jpg',
                                    title = 'Raw tweets per user per day',
                                    variable = 'raw_tweet',
                                    y_max = 35,
                                    list = [0, 5, 10, 15, 20, 25, 30])

def get_network_stat():

    df = import_data('twitter_data_climate_tweets_2022_03_15.csv')

    df_type = import_data('type_users_climate.csv')
    list_d = df_type[df_type['type'] == 1 ]['username'].tolist()
    list_s = df_type[df_type['type'] == 2 ]['username'].tolist()
    list_a = df_type[df_type['type'] == 4]['username'].tolist()

    #df = df[df['username'].isin(list_users)]
    print(df.columns)


    df_rt = df.groupby(['retweeted_username_within_list'], as_index = False).size().sort_values(by = 'size', ascending = False)
    df_qt = df.groupby(['quoted_username_within_list'], as_index = False).size().sort_values(by = 'size', ascending = False)
    df_rp = df.groupby(['in_reply_to_username_within_list'], as_index = False).size().sort_values(by = 'size', ascending = False)

    print( 100*(df_rt['size'].sum() / df['retweeted_username'].count()))
    print( 100*(df_qt['size'].sum() / df['quoted_username'].count()))
    print( 100*(df_rp['size'].sum() / df['in_reply_to_username'].count()))

    print(df[df['username'].isin(list_d)]['retweeted_username_within_list'].count()/df[df['username'].isin(list_d)]['retweeted_username'].count())
    print(df[df['username'].isin(list_d)]['quoted_username_within_list'].count()/df[df['username'].isin(list_d)]['quoted_username'].count())
    print(df[df['username'].isin(list_d)]['in_reply_to_username_within_list'].count()/df[df['username'].isin(list_d)]['in_reply_to_username'].count())

    print(df[df['username'].isin(list_d)]['retweeted_username_within_list'].count()/df[df['username'].isin(list_d)]['retweeted_username'].count())
    print(df[df['username'].isin(list_d)]['quoted_username_within_list'].count()/df[df['username'].isin(list_d)]['quoted_username'].count())
    print(df[df['username'].isin(list_d)]['in_reply_to_username_within_list'].count()/df[df['username'].isin(list_d)]['in_reply_to_username'].count())

def main():

    create_general_activity_figures()

if __name__ == '__main__':

    main()
