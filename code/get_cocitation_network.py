import ast
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import prince
import plotly.express as px
import numpy as np
import time
import networkx as nx
import networkx.algorithms.community as nxcom

from functools import reduce
from collections import Counter
from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from ural import get_domain_name
from utils import (import_data,
                    save_data,
                    save_figure,
                    save_numpy_array,
                    save_list,
                    tic,
                    toc)

from create_twitter_users_lists import get_lists_and_followers

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def add_type(var1, var2, df):

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()

    list_manual = [
    'israhirsi',
    'xiuhtezcatl',#not found in tweets cop26 artist
    'lillyspickup',
    'jamie_margolin', #not found in tweets cop26
    'nakabuyehildaf', #not found in tweets cop26
    'namugerwaleah',#not found in tweets cop26
    'anunade', #german description but tweeted during COP26
    'varshprakash',#not found in tweets cop26
    'jeromefosterii'
    ]

    df = df[~df['username'].isin(list_manual)]


    df[var1] = ''
    df[var1] = np.where(df[var2].isin(list_scientists), 'scientist', df[var1])
    df[var1] = np.where(df[var2].isin(list_activists), 'activist', df[var1])
    df[var1] = np.where(df[var2].isin(list_delayers), 'delayer', df[var1])

    return df

def intersection(lst1, lst2):

    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def get_tweets():

    df = import_data ('twitter_data_climate_tweets_2022_03_15.csv')
    df = df[~df['query'].isin(['f'])]
    df = add_type('type', 'username', df)
    df['hashtags'] = df['hashtags'].str.lower()
    #print('number of tweets', len(df))

    return df

def get_list_urls(df, var1, var2, type):

    i = df.index[df[var1] == type]
    list_urls = df[var2].iloc[i]
    print(type(list_urls))

    return list_urls

def get_common_domains():

    df = get_tweets()

    df = df[['username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'type']]
    #df = df[~df['type_of_tweet'].isin(['replied_to'])]
    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])
    #print('after removing replies', len(df))
    df = df.explode('domain_name')
    df = df.dropna(subset=['domain_name'])

    url_scientists = df[df['type'].isin(['scientist'])]['domain_name'].unique().tolist()
    url_activists = df[df['type'].isin(['activist'])]['domain_name'].unique().tolist()
    url_delayers = df[df['type'].isin(['delayers'])]['domain_name'].unique().tolist()

    list_1 = [x for x in url_scientists if x in url_activists]
    print(list_1)
    print(len(list_1))

    list_all = [x for x in url_delayers if x in list_1]
    print(len(list_all))
    print(list_all)


def aggregate_domains_per_user():

    df = get_tweets()

    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])

    u = df.groupby(['username', 'type'])['domain_name'].apply(list).reset_index(name='list_domain_names')
    u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list({x for l in list_items for x in l}))
    #with repetitions
    #u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list(x for l in list_items for x in l))
    list_platforms = ['twitter.com', 'youtube.com', 'bit.ly', 'google.com', 'yahoo.com']

    u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list(x for x in list_items if x not in list_platforms))
    u['len_list'] = u['list_domain_names'].apply(len)
    u = u.sort_values(by = 'type')
    print(u.head())
    print(u.tail())

    #print(u['len_list'].describe())

    return u

def get_cocitation(limit_cocitations):

    timestr = time.strftime('%Y_%m_%d')
    df = aggregate_domains_per_user()

    list_individuals = df['username'].tolist()
    save_list(list_individuals, 'list_individuals_cocitations.txt')
    #print(list_individuals[0:10])
    n = len(list_individuals)
    #print('number of individuals', n)
    matrix = np.zeros((n,n))
    matrix_lim = np.zeros((n,n))

    for user_i in list_individuals:
        #print(user_i)
        for user_j in list_individuals:
            if user_i != user_j:
                i = df.index[df['username'] == user_i]
                j = df.index[df['username'] == user_j]
                a = intersection(df['list_domain_names'].iloc[i[0]], df['list_domain_names'].iloc[j[0]])
                matrix[i,j] = len(a)
                matrix[i,i] = len(df['list_domain_names'].iloc[i[0]])

                matrix_lim[i,i] = 0

                if matrix[i,j] > limit_cocitations:
                    matrix_lim[i,j] = 1
                else:
                    matrix_lim[i,j] = 0

    save_numpy_array(matrix, 'cocitations_{}.npy'.format(timestr))
    #print(matrix)
    #print(matrix_lim[1,:])
    s = np.sum(matrix_lim, axis=1)
    G = nx.from_numpy_matrix(matrix_lim)
    #print(G.nodes)
    for index, row in df.iterrows():
        G.nodes[index]['type'] = row['type']
        G.nodes[index]['username'] = row['username']

    nx.write_gexf(G, './data/{}_network_climate_cocitations_{}.gexf'.format(limit_cocitations, timestr))
    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    l_com = len(communities)
    print('communities', l_com, 'lim cocitation', limit_cocitations)
    #print(G.nodes[0]['type'])
    n_zeros = np.count_nonzero(s==0)
    #print(s)
    #print(len(s))
    #print(n_zeros)

    return l_com

def to_1D(series):

 return pd.Series([x for _list in series for x in _list], name = 'hashtag_count')

def get_hashtags(limit_occurence):

    df = get_tweets()
    df['hashtags'] = df['hashtags'].apply(eval)
    df['len_hashtags']= df['hashtags'].apply(len)

    series = to_1D(df['hashtags']).value_counts()
    df1 = series.to_frame()
    print(df1['hashtag_count'].describe())
    df1.index.name='hashtags'
    #df1['hashtags'] = df1['hashtags'].str.lower()
    df1 = df1.reset_index(level=0)
    df1 = df1[df1['hashtag_count']> limit_occurence]
    print(df1['hashtags'].head(20))
    return df1

def get_hashtags_by_type() :

    df = get_tweets()

    df = df[['username', 'hashtags', 'type_of_tweet', 'id', 'text', 'followers_count', 'type']]

    a = len(df[df['type'].isin(['activist'])])
    b = len(df[df['type'].isin(['delayer'])])
    c = len(df[df['type'].isin(['scientist'])])


    #df = df[~df['type_of_tweet'].isin(['replied_to'])]
    for index, row in df.iterrows():
        df.at[index, 'hashtags']=ast.literal_eval(row['hashtags'])

    df['nb_hashtags'] = df['hashtags'].apply(len)
    print(df['nb_hashtags'].head(20))

    print('number of tw with hashtags', len(df[df['nb_hashtags']>0]))
    df = df.explode('hashtags')

    df = df.dropna(subset=['hashtags'])
    print(df.head(40))
    print('There are', df['hashtags'].nunique(), 'unique hastag')
    print(df.groupby(['type'], as_index = False).size())
    df1 = df[df['nb_hashtags']>0].groupby(['type'], as_index = False).size()
    df1['share_tw_hashtags'] = 0
    df1['share_tw_hashtags'].iloc[0] =  df1['size'].iloc[0]/a
    df1['share_tw_hashtags'].iloc[1] =  df1['size'].iloc[1]/b
    df1['share_tw_hashtags'].iloc[2] =  df1['size'].iloc[2]/c
    print(df1)
    print(df[df['nb_hashtags']>0].groupby(['type'], as_index = False).size())

    return df

def main():

    df = pd.DataFrame(columns=['lim_cocitations',
                               'nb_communities'])

    for i in range(0,200,10):
        l_com = get_cocitation(limit_cocitations = i)
        df = df.append({'lim_cocitations': i,
                        'nb_communities': l_com}, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    save_data(df, 'communities_{}.csv'.format(timestr), 0)
    #get_hashtags(limit_occurence = 10)
    #get_hashtags_by_type()

def get_shares_types(cocitation_lim, timestr):

    G = nx.read_gexf('./data/{}_network_climate_cocitations_{}.gexf'.format(cocitation_lim, timestr))
    #type = nx.get_node_attributes(G, "type")
    #print(G.nodes)
    #print(G.nodes(data=True))
    #list = G.nodes(data=True)
    #print(list(G.nodes(data=True))[0:10])
    #print(G.nodes['1']['type'])
    #print(list(G.neighbors('3')))
    df_share_cocitaion = pd.DataFrame(columns=[ 'username',
                                               'own_type',
                                               'number_of_neighbors',
                                               'share_cocitation_scientist',
                                               'share_cocitation_activist',
                                               'share_cocitation_delayer'])
    for i in G.nodes:
        neighbors_i = list(G.neighbors(i))
        n = len(list(G.neighbors(i)))
        list_neighbor_type = []

        for j in neighbors_i:
            neighbor_type = G.nodes[j]['type']
            list_neighbor_type.append(neighbor_type)

        if 'scientist' in Counter(list_neighbor_type).keys():
            a = round((Counter(list_neighbor_type)['scientist'])/n, 2)
        else:
            a = 0

        if 'activist' in Counter(list_neighbor_type).keys():
            b = round((Counter(list_neighbor_type)['activist'])/n, 2)
        else:
            b = 0

        if 'delayer' in Counter(list_neighbor_type).keys():
            c = round((Counter(list_neighbor_type)['delayer'])/n, 2)
        else:
            c = 0

        df_share_cocitaion = df_share_cocitaion.append({ 'username': G.nodes[i]['username'],
                                                        'own_type': G.nodes[i]['type'],
                                                        'number_of_neighbors': n,
                                                        'share_cocitation_scientist': a,
                                                        'share_cocitation_activist': b,
                                                        'share_cocitation_delayer': c },
                                                        ignore_index=True)

    save_data(df_share_cocitaion, 'share_cocitation_{}_by_type.csv'.format(cocitation_lim), 0)

    # neighbors_ajw = list(G.neighbors('17'))
    #
    # list_neighbor_type = []
    # for i in neighbors_ajw:
    #     neighbor_type = G.nodes[i]['type']
    #     list_neighbor_type.append(neighbor_type)
    #
    #
    # print('type of ind', G.nodes['17']['type'])
    # print(Counter(list_neighbor_type).keys())
    # print(Counter(list_neighbor_type).values())
    # print(Counter(list_neighbor_type)['scientist'])

if __name__ == '__main__':

    #get_cocitation(limit_cocitations = 5)
    #get_shares_types(cocitation_lim = 10, timestr = '2022_07_20')
    #aggregate_domains_per_user()
    get_common_domains()
