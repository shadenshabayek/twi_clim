import ast
import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None  # default='warn'

from matplotlib import pyplot as plt
from pandas.api.types import CategoricalDtype
from ural import get_domain_name
from utils import (import_data,
                    import_google_sheet,
                    push_to_google_sheet,
                    save_data,
                    save_figure)
from create_twitter_users_lists import get_lists_and_followers


"""create initial dataset to be imported to a google spreadsheet for human annotation"""

def add_type(var1, var2, df):

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()
    df[var1] = ''
    df[var1] = np.where(df[var2].isin(list_scientists), 'scientist', df[var1])
    df[var1] = np.where(df[var2].isin(list_activists), 'activist', df[var1])
    df[var1] = np.where(df[var2].isin(list_delayers), 'delayer', df[var1])

    return df

def get_domain_names_Twitter (type):

    df = import_data ('twitter_data_climate_tweets_2022_03_15.csv')
    df = add_type('type', 'username', df)
    print('number of tweets', len(df))
    #print(df.columns)

    if type == 'scientist':
        df = df[df['type'].isin(['scientist'])]
    elif type == 'activist':
        df = df[df['type'].isin(['activist'])]
    elif type == 'delayer':
        df = df[df['type'].isin(['delayer'])]

    df['positive_engagement'] = df['retweet_count'] + df['like_count']

    df = df.dropna(subset=['domain_name'])
    df = df[['username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'text', 'positive_engagement']]

    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])

    df=df.explode('domain_name')

    a = ['twitter.com']
    print('number of tweets containing a tw link', df[df['domain_name'].isin(a)]['domain_name'].count())

    df = df[~df['domain_name'].isin(a)]
    df = df.dropna(subset=['domain_name'])
    df['username'] = df['username'].str.lower()
    print('number of tweets containing a link excluding Tw', df['expanded_urls'].count())
    print('There are', df['domain_name'].nunique(), 'unique domain names, out of', df['domain_name'].count())

    return df

def get_domains_ratings (type):

    rating = 'third_aggregation'

    df1 = import_google_sheet ('domain_names_rating')
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df_ratings = get_domain_names_Twitter (type)

    df_ratings[rating] = ''

    df_ratings.set_index('domain_name', inplace=True)
    df_ratings.update(df1.set_index('domain_name'))
    df_ratings=df_ratings.reset_index()

    df_ratings[rating] = df_ratings[rating].replace('','unrated')
    summary_ratings = df_ratings.groupby([rating], as_index = False).size().sort_values(by = 'size', ascending = False)
    print(summary_ratings)

    total_number_of_links = len(df_ratings)
    print('Number of links contained in tweets, with repetitions', total_number_of_links)

    df_unrated = df_ratings[df_ratings[rating] == 'unrated']
    #print(df_ratings.info())
    #print(df_ratings['id'].nunique())
    print(df_ratings.groupby(['third_aggregation'], as_index = False)['positive_engagement'].mean())

    return df_ratings, df_unrated, total_number_of_links, summary_ratings

def get_domains_categories (type):

    df_ratings, df_unrated, total_number_of_links, summary_ratings = get_domains_ratings (type)

    df1 = import_google_sheet ('domain_names_rating')
    #print('number of unique domain names', df1['domain_name'].nunique())
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df_unrated['category']=''

    df_unrated.set_index('domain_name', inplace=True)
    df_unrated.update(df1.set_index('domain_name'))
    df_unrated=df_unrated.reset_index()

    df_unrated['category'] = df_unrated['category'].replace('','uncategorized')

    #remove = ['uncategorized']
    #df2 = df2[~df2['category'].isin(remove)]
    print('number of unrated links with repetition', len(df_unrated))
    summary_categories_unrated = df_unrated.groupby(['category'], as_index = False).size().sort_values(by = 'size', ascending = False)
    print(summary_categories_unrated)
    print( 'share of uncategorized and unrated', len(df_unrated[df_unrated['category'] == 'uncategorized'])/total_number_of_links)
    print('unique links', df_unrated[df_unrated['category'] == 'uncategorized']['domain_name'].nunique() )
    #print(df_unrated[df_unrated['category'] == 'uncategorized'].groupby(['domain_name'], as_index = False).size().sort_values(by = 'size', ascending = False).head(50))

    return df_unrated, summary_categories_unrated, summary_ratings

def create_donut(x, y, df, figure_name, title, type_title, colormap):

    fig, ax = plt.subplots(figsize=(6, 15), subplot_kw=dict(aspect="equal"))

    df = df

    l = len(df[x].to_list())
    list_labels =[]
    for i in range(0,l):
        a = df[x].to_list()[i] + ' (' + str(df[y].to_list()[i]) + ')'
        list_labels.append(a)

    ratings = list_labels
    data = df[y].to_list()

    #cmap = plt.get_cmap('cool')
    #cmap = plt.get_cmap('Greens')
    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, 7)]

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.4), startangle=230, colors = colors)

    bbox_props = dict(boxstyle="square,pad=0.2", fc="w", ec="k", lw=0.72)

    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    plt.text(0, 0, '{}'.format(type_title), ha='center', va='center', fontsize=14)

    for i, p in enumerate(wedges):

        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(ratings[i], xy=(x, y), xytext=(1.3*np.sign(x), 1.3*y),
                    horizontalalignment=horizontalalignment, **kw)

    save_figure(figure_name)

def create_donut_by_group(type, type_title, colormap, type_df):

    df_unrated, summary_categories_unrated, summary_ratings = get_domains_categories (type = type)

    if type_df == 'rating':
        create_donut(x = 'third_aggregation',
                          y = 'size',
                          df = summary_ratings,
                          figure_name = 'summary_ratings_climate_{}'.format(type),
                          title = '',
                          type_title = type_title,
                          colormap = colormap)

    elif type_df == 'category':
        create_donut(x = 'category',
                          y = 'size',
                          df = summary_categories_unrated,
                          figure_name = 'summary_categories_climate_{}'.format(type),
                          title = '',
                          type_title = type_title,
                          colormap = colormap)

def create_figures():

    create_donut_by_group(type = 'scientist',
                            type_title = 'Scientists',
                            colormap = 'Greens',
                            type_df = 'category')

    create_donut_by_group(type = 'activist',
                            type_title = 'Activists',
                            colormap = 'Oranges',
                            type_df = 'category')

    create_donut_by_group(type = 'delayer',
                            type_title = 'Delayers',
                            colormap = 'Reds',
                            type_df = 'category')

    create_donut_by_group(type = 'all',
                            type_title = 'All groups',
                            colormap = 'cool',
                            type_df = 'category')

if __name__ == '__main__':

    #create_figures()
    get_domains_ratings (type = 'all')
    get_domains_ratings (type = 'scientist')
    get_domains_ratings (type = 'activist')
    get_domains_ratings (type = 'delayer')
