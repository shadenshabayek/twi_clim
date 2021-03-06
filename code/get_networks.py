import ast
from datetime import date
import lxml.etree as etree
import pandas as pd
import numpy as np
import os
import pickle
import time
import json_lines
import networkx as nx

from matplotlib import pyplot as plt

from create_twitter_users_lists import get_lists_and_followers

from utils import (import_data,
                    import_google_sheet,
                    save_data,
                    save_figure,
                    save_list,
                    read_list,
                    save_numpy_array,
                    read_numpy_array)

def add_type(var1, var2, df):

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()
    df[var1] = ''
    df[var1] = np.where(df[var2].isin(list_scientists), 'scientist', df[var1])
    df[var1] = np.where(df[var2].isin(list_activists), 'activist', df[var1])
    df[var1] = np.where(df[var2].isin(list_delayers), 'delayer', df[var1])

    return df

def get_tweets_by_type():

    df  = import_data('twitter_data_climate_tweets_2022_03_15.csv')
    df = df[~df['query'].isin(['f'])]

    df = add_type('type', 'username', df)
    df = add_type('type_retweeted', 'retweeted_username_within_list', df)
    df = add_type('type_quoted', 'quoted_username_within_list', df)
    df = add_type('type_in_reply', 'in_reply_to_username_within_list', df)

    print(df[~df['type'].isin(['scientist', 'activist','delayer'])]['username'].unique())

    return df

def create_gexf(var1, var2, type_tweet):

    timestr = time.strftime("%Y_%m_%d")
    df = get_tweets_by_type()
    #they have no ratings, so for score, absorbing share_alternative_paltforms
    list_drop = ['sumakhelena', 'yann_a_b', 'israhirsi', 'johnredwood', 'weatherdotus']
    df = df[~df[var1].isin(list_drop)]
    df = df[df[var1].notna()]
    df = df[df['username'] != df[var1]]
    df1 = df[['username', 'type', var1, var2]]
    save_data(df1, '{}_network_{}.csv'.format(type_tweet, timestr), 0)
    G = nx.Graph()

    G = nx.from_pandas_edgelist(df, 'username', var1, create_using=nx.MultiGraph())

    for index, row in df.iterrows():
        G.nodes[row[var1]]['type'] = row[var2]
        G.nodes[row['username']]['type'] = row['type']

    nx.write_gexf(G, "./data/{}_network_climate_{}.gexf".format(type_tweet, timestr))

    list_nodes = G.nodes
    A = nx.adjacency_matrix(G)
    B = nx.to_numpy_array(G)

    save_list(list_nodes, 'list_nodes_climate_{}_{}.txt'.format(type_tweet, timestr))
    save_numpy_array(B, 'matrix_climate_{}_{}.npy'.format(type_tweet, timestr))

    return A, B, list_nodes

def assign_color_by_type(type_user):

    if type_user == 'scientist' :
        color = 'green'
    elif type_user == 'activist' :
        color = 'darkorange'
    elif type_user == 'delayer' :
        color = 'red'

    return color

def update_score(type_tweet):

    timestr = time.strftime("%Y_%m_%d")
    #timestr = '2022_04_25'
    title = 'climate_percentage_rating_agg_' + timestr + '.csv'

    df_initial_score = import_data(title)

    list_nodes = read_list('list_nodes_climate_{}_{}.txt'.format(type_tweet, timestr))
    print(list_nodes)
    B = read_numpy_array('matrix_climate_{}_{}.npy'.format(type_tweet, timestr))

    df_users_scores = pd.DataFrame(list_nodes, columns=['username'])
    df = df_users_scores.merge(df_initial_score, how = 'inner', on = ['username'])
    print('number of nodes', len(list_nodes))

    score = df['share_negative'].to_numpy()

    print(score.shape)
    print(B.shape)

    np.fill_diagonal(B, 1, wrap=False)
    B_w = B/B.sum(axis=1, keepdims=True)

    print((B_w))

    n = len(df['username'].tolist())
    N = 300
    matrix = np.zeros((n,N))
    matrix[:,0] = score
    #print(matrix)

    plt.figure(figsize=(10, 6))
    ax = plt.subplot(111)

    for i in range(1,N):

        for user in range(0,len(df['username'].tolist())):

            a = (B_w[user][:]).dot(matrix[:,(i-1)])
            matrix[user,i] = a

            color = assign_color_by_type(df['type'].iloc[user])

            ax.plot(i,
                     a,
                     marker = 'o',
                     color = color,
                     linestyle='solid',
                     linewidth = 1
                     )
    print(matrix[:,N-1])
    df['final_score'] = matrix[:,N-1]
    plt.ylim(-3.5, 100)
    plt.xlim(-0.1,N)
    #plt.xlabel(xlabel, size='large')

    plt.tight_layout()
    #save_figure('score_updating_based_on_{}_test_share_neg.jpg'.format(type_tweet))
    #save_data(df, 'final_score_unweighted_eng.csv', 0)
    return df

def create_networks():

    create_gexf(var1 = 'retweeted_username_within_list',
                var2 = 'type_retweeted',
                type_tweet = 'retweeted')

    create_gexf(var1 = 'quoted_username_within_list',
                var2 = 'type_quoted',
                type_tweet = 'quoted')

    create_gexf(var1 = 'in_reply_to_username_within_list',
                var2 = 'type_in_reply',
                type_tweet = 'reply')

def plot_score_update():

    create_gexf(var1 = 'retweeted_username_within_list',
                var2 = 'type_retweeted',
                type_tweet = 'retweeted')

    update_score(type_tweet = 'retweeted')

def main():

    create_networks()
    plot_score_update()

if __name__ == '__main__':

    main()
