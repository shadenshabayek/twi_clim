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
                    save_figure)

def save_list(list, file_name):

    file_path = os.path.join('.', 'data', file_name)
    #file_name has to be .txt

    with open(file_path, "wb") as fp:
        pickle.dump(list, fp)

def read_list(file_name):

    file_path = os.path.join('.', 'data', file_name)

    with open(file_path, "rb") as fp:   # Unpickling
        b = pickle.load(fp)

    return b

def save_numpy_array(array, file_name):

    file_path = os.path.join('.', 'data', file_name)
    #filename has to be .npy
    with open(file_path, 'wb') as f:
        np.save(f, array)

def read_numpy_array(file_name):
    file_path = os.path.join('.', 'data', file_name)
    #filename has to be .npy
    with open(file_path, 'rb') as f:

        a = np.load(f)

    return a

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

    df = get_tweets_by_type()
    df = df[df[var1].notna()]
    df = df[df['username'] != df[var1]]
    df1 = df[['username', 'type', var1, var2]]
    save_data(df1, '{}_network.csv'.format(type_tweet), 0)
    G = nx.Graph()

    G = nx.from_pandas_edgelist(df, 'username', var1, create_using=nx.MultiGraph())

    for index, row in df.iterrows():
        G.nodes[row[var1]]['type'] = row[var2]
        G.nodes[row['username']]['type'] = row['type']

    nx.write_gexf(G, "./data/{}_network_climate.gexf".format(type_tweet))

    #print('number of edges', G.number_of_edges())
    #print('number of nodes', G.number_of_nodes())
    #print(G.nodes)
    #list_nodes = G.nodes
    #A = nx.adjacency_matrix(G)
    #B = nx.to_numpy_array(G)
    #print(B.shape)
    #print(len(list_nodes))

    #save_list(list_nodes, 'list_nodes_climate_retweets.txt')
    #save_numpy_array(B, 'matrix_mentions_retweets.npy')

    #return A, B, list_nodes

def assign_color_by_type(type_user):

    if type_user == 1 :
        color='red'
    elif type_user == 2 :
        color='green'
    elif type_user == 31 :
        color='lightcoral'
    elif type_user == 32 :
        color='lightgreen'
    elif type_user == 312 :
        color='deepskyblue'
    elif type_user == 4 :
        color='darkorange'

    return color

def update_score():

    df_initial_score = import_data('climate_score_postives_count.csv')
    df_initial_score = df_initial_score.rename(columns = {'Twitter_handle': 'username'})
    df_initial_score = df_initial_score.drop(['own_website',
                                                'other_link_count_with_rating',
                                                'own_link_count_with_rating'], axis = 1)


    #A, B, list_nodes = create_mentions_gexf()

    list_nodes = read_list('list_nodes_climate_retweets.txt')
    B = read_numpy_array('matrix_mentions_retweets.npy')

    df_users_scores = pd.DataFrame(list_nodes, columns=['username'])

    df = df_users_scores.merge(df_initial_score, how = 'inner', on = ['username'])
    df['type'] = df['type'].astype('int')

    print('number of nodes', len(list_nodes))

    score = df['final_score_twitter'].to_numpy()

    print(score.shape)
    print(B.shape)

    B_w = B/B.sum(axis=1, keepdims=True)

    n = len(df['username'].tolist())
    N = 100
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

            # ax.scatter(i,
            #          a,
            #          color = color,
            #          linestyle='solid',
            #          linewidth = 3
            #          )
            ax.plot(i,
                     a,
                     marker = 'o',
                     color = color,
                     linestyle='solid',
                     linewidth = 3
                     )

    plt.ylim(-3.5, 3.5)
    plt.xlim(-0.1,N)
    #plt.xlabel(xlabel, size='large')

    plt.tight_layout()
    save_figure('score_updating_based_on_retweets.jpg')

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

if __name__ == '__main__':

    #create_mentions_gexf()
    #update_score()
    create_networks()
