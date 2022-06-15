import os
from datetime import datetime, timedelta, date
from utils import (import_data, save_figure)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #new addition to create the boxes in the legend for the shaded areas
import ural

def plot_format(ax, plt):

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.locator_params(axis='y', nbins=4)
    ax.xaxis.set_tick_params(length=0)

    ax.grid(axis='y')
    handles, labels = ax.get_legend_handles_labels()

    plt.legend(handles=handles, loc = 'upper left')

    #plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

def get_df_by_group(df, type, variable):

    df_type = import_data('type_users_climate.csv')
    list_users = df_type[df_type['type'] == type ]['username'].tolist()

    df = df[df['username'].isin(list_users)]
    df['type_of_tweet'] = df['type_of_tweet'].replace(np.nan, 'created_content')
    df['total_engagement'] = (df['retweet_count'] + df['like_count'] + df['reply_count'])
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    #print(df.groupby(['type_of_tweet'], as_index = False).size())
    df1 = df[df['type_of_tweet'] == variable].groupby(['date', 'username'], as_index=False).size()
    print(df1.head(20))
    df_mean = df1.groupby(['date'], as_index=False)['size'].mean()
    print(df_mean)

    return df_mean

def create_twitter_figure_per_user(filename, figure_name, title, variable, y_max, list):

    df = import_data(filename)
    df = df.drop_duplicates()

    df_d = get_df_by_group(df = df, type = 1, variable = variable)
    df_s = get_df_by_group(df = df, type = 2, variable = variable)
    df_a = get_df_by_group(df = df, type = 4, variable = variable)
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

    ax.set_xlim([date(2021, 6, 2), date(2021, 11, 29)])

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

    plt.text(np.datetime64("2021-11-02"), 1, "COP26", fontsize=7, color='g')

    plot_format(ax, plt)

    save_figure(figure_name)
    #plt.show()

def create_twitter_figure(filename, figure_name, title, type):

    df = import_data(filename)
    df = df.drop_duplicates()

    df_type = import_data('type_users_climate.csv')
    list_users = df_type[df_type['type'] == type ]['username'].tolist()

    df = df[df['username'].isin(list_users)]
    df['type_of_tweet'] = df['type_of_tweet'].replace(np.nan, 'created_content')
    df['total_engagement'] = (df['retweet_count'] + df['like_count'] + df['reply_count'])
    print(df['created_at'].iloc[0])
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    print(df.groupby(['type_of_tweet'], as_index = False).size())
    df_rt = df[df['type_of_tweet'] == 'retweeted'].groupby(['date', 'username'], as_index=False).size()
    print(df_rt.head(20))

    df_rp = df[df['type_of_tweet'] == 'replied_to'].groupby(['date'], as_index=False).size()
    df_qt = df[df['type_of_tweet'] == 'quoted'].groupby(['date'], as_index=False).size()
    df_cc = df[df['type_of_tweet'] == 'created_content'].groupby(['date'], as_index=False).size()

    fig, ax = plt.subplots(figsize=(8, 4))

    d = df[(df['date']> date(2019, 1, 1) ) & (df['date']<date(2021, 7, 1))]
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

    ax.set_xlim([date(2021, 6, 2), date(2021, 11, 29)])
    ax.set_ylim([-0.1, 2100])
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

    plot_format(ax, plt)

    save_figure(figure_name)
    #plt.show()

def create_general_activity_figures():

        create_twitter_figure(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                            figure_name = 'twitter_volume_climate_scientists.jpg',
                            title = 'Scientists',
                            type = 2 )

        create_twitter_figure(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                            figure_name = 'twitter_volume_climate_activists.jpg',
                            title = 'Activists',
                            type = 4 )
        create_twitter_figure(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                            figure_name = 'twitter_volume_climate_delayer.jpg',
                            title = 'Delayers',
                            type = 1 )

        create_twitter_figure_per_user(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                                        figure_name = 'retweets_per_day.jpg',
                                        title = 'Retweets per user per day',
                                        variable = 'retweeted',
                                        y_max = 35,
                                        list = [0, 5, 10, 15, 20, 25, 30])

        create_twitter_figure_per_user(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                                            figure_name = 'replies_per_day.jpg',
                                            title = 'Replies per user per day',
                                            variable = 'replied_to',
                                            y_max = 35,
                                            list = [0, 5, 10, 15, 20, 25, 30])

        create_twitter_figure_per_user(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                                                figure_name = 'quotes_per_day.jpg',
                                                title = 'Quotes per user per day',
                                                variable = 'quoted',
                                                y_max = 35,
                                                list = [0, 5, 10, 15, 20, 25, 30])

        create_twitter_figure_per_user(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                                        figure_name = 'cc_per_day.jpg',
                                        title = 'Raw tweets per user per day',
                                        variable = 'created_content',
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
