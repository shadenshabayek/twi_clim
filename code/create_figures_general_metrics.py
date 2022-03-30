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

    # if suspension == 1 :
    #     patch = mpatches.Patch(color='pink',
    #                            label='Suspension Period')
    #     handles.append(patch)

    plt.legend(handles=handles)

    plt.setp(ax.get_xticklabels(), rotation=45)

    plt.tight_layout()

def create_twitter_figure(filename, figure_name, title):

    df = import_data(filename)
    df = df.drop_duplicates()

    df['type_of_tweet'] = df['type_of_tweet'].replace(np.nan, 'created_content')
    df['total_engagement'] = (df['retweet_count'] + df['like_count'] + df['reply_count'])
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    #df_volume = df.groupby(['date','type_of_tweet'], as_index=False).size()
    df_volume = df.groupby(['date'], as_index=False).size()


    fig, ax = plt.subplots(figsize=(10, 4))

    #d = df[(df['date']> date(2019, 1, 1) ) & (df['date']<date(2021, 7, 1))]
    #total = d['id'].count()

    ax.plot(df_volume['date'],
        df_volume['size'],
        color='deepskyblue',
        label='Number of Tweets per day')

    ax.set(
       title = title )

    ax.set_xlim([date(2021, 6, 2), date(2021, 11, 29)])

    # plt.axvspan(np.datetime64('2019-12-09'),
    #             np.datetime64('2020-10-12'),
    #             ymin=0, ymax=200000,
    #             facecolor='r',
    #             alpha=0.05)
    #
    # plt.axvspan(np.datetime64('2021-01-25'),
    #             np.datetime64('2021-06-30'),
    #             ymin=0,
    #             ymax=200000,
    #             facecolor='r',
    #             alpha=0.05)

    plot_format(ax, plt)

    save_figure(figure_name)
    #plt.show()

if __name__ == '__main__':

    create_twitter_figure(filename = 'twitter_data_climate_tweets_2022_03_15.csv',
                        figure_name = 'twitter_volume_climate.jpg',
                        title = 'twitter_volume')
