import ast
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import prince
import plotly.express as px
import numpy as np
import time
import networkx as nx


from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from ural import get_domain_name
from utils import (import_data,
                    save_data,
                    save_figure,
                    save_numpy_array,
                    save_list)

from create_twitter_users_lists import get_lists_and_followers

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def add_type(var1, var2, df):

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()
    df[var1] = ''
    df[var1] = np.where(df[var2].isin(list_scientists), 'scientist', df[var1])
    df[var1] = np.where(df[var2].isin(list_activists), 'activist', df[var1])
    df[var1] = np.where(df[var2].isin(list_delayers), 'delayer', df[var1])

    return df

def intersection(lst1, lst2):

    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def aggregate_domains_per_user():

    df = import_data ('twitter_data_climate_tweets_2022_03_15.csv')
    df = df[~df['query'].isin(['f'])]
    df = add_type('type', 'username', df)
    print('number of tweets', len(df))

    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])

    u = df.groupby(["username", "type"])["domain_name"].apply(list).reset_index(name='list_domain_names')
    #u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list({x for l in list_items for x in l}))
    u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list(x for l in list_items for x in l))
    list_platforms = ['twitter.com', 'youtube.com', 'bit.ly', ]
    u['list_domain_names'] = u['list_domain_names'].apply(lambda list_items: list(x for x in list_items if x not in list_platforms))
    u['len_list'] = u['list_domain_names'].apply(len)
    u = u.sort_values(by = 'type')
    #print(u.head(50))
    print(u.info())
    print(u['len_list'].describe())
    #a= u['list_domain_names'].iloc[85]
    #print(u['list_domain_names'].iloc[85])
    #print(len(u['list_domain_names'].iloc[85]))
    return u

def get_cocitation(lim):

    timestr = time.strftime("%Y_%m_%d")
    df = aggregate_domains_per_user()

    list_individuals = df['username'].tolist()
    save_list(list_individuals, 'list_individuals_cocitations.txt')
    print(list_individuals[0:10])
    n = len(list_individuals)
    print('number of individuals', n)
    matrix = np.zeros((n,n))
    matrix_lim = np.zeros((n,n))

    for user_i in list_individuals:
        print(user_i)
        for user_j in list_individuals:
            if user_i != user_j:
                i = df.index[df['username'] == user_i]
                j = df.index[df['username'] == user_j]
                a = intersection(df['list_domain_names'].iloc[i[0]], df['list_domain_names'].iloc[j[0]])
                matrix[i,j] = len(a)
                matrix[i,i] = len(df['list_domain_names'].iloc[i[0]])

                matrix_lim[i,i] = 0

                if matrix[i,j] > lim:
                    matrix_lim[i,j] = 1
                else:
                    matrix_lim[i,j] = 0

    save_numpy_array(matrix, 'cocitations_{}.npy'.format(timestr))
    #print(matrix)
    #print(matrix_lim[1,:])
    s = np.sum(matrix_lim, axis=1)
    G = nx.from_numpy_matrix(matrix_lim)
    print(G.nodes)
    for index, row in df.iterrows():
        G.nodes[index]['type'] = row['type']
        G.nodes[index]['username'] = row['username']


    nx.write_gexf(G, "./data/{}_network_climate_cocitations_{}.gexf".format(lim, timestr))
    #print(G.nodes[0]['type'])
    n_zeros = np.count_nonzero(s==0)
    print(s)
    print(len(s))
    print(n_zeros)

    return matrix


if __name__ == '__main__':
    #get_total_citations_by_user()
    #aggregate_domains_per_user()
    get_cocitation(lim = 100)
