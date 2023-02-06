import ast
import pandas as pd
import numpy as np
import time

pd.options.mode.chained_assignment = None  # default='warn'
from matplotlib.ticker import StrMethodFormatter
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

    df = df[~df['query'].isin(['f'])]
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
    df = df[['type', 'username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'text', 'positive_engagement']]

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
    df_freq = df.groupby(['domain_name'], as_index = False).size().sort_values(by = 'size', ascending = False)
    print(df_freq.describe())
    print(df_freq.head(50))

    return df, df_freq

def plot_hist(df, var):

    total = df['size'].sum()
    print(total)

    df = df[df[var] > 800]
    sub_total = df['size'].sum()
    print(sub_total)

    share = round(100*(sub_total/total))
    print( 'share of top 1000 out of total', share)
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_axes([0,0,1,1])
    ax.bar(df['domain_name'], df['size'])

    plt.xticks(rotation = 30, fontsize = 12, ha = 'right')
    plt.yticks(np.arange(0, max(df['size']), 500))

    save_figure('distribution_links')

def get_domains_ratings (type):

    rating = 'MBFC_factual'

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
    unrated_uncategorized = len(df_unrated[df_unrated['category'] == 'uncategorized'])
    unrated_categorized = len(df_unrated) - unrated_uncategorized
    print( 'share of uncategorized and unrated', len(df_unrated[df_unrated['category'] == 'uncategorized'])/total_number_of_links)
    print('unique links', df_unrated[df_unrated['category'] == 'uncategorized']['domain_name'].nunique() )
    print(df_unrated[df_unrated['category'] == 'uncategorized'].groupby(['domain_name'], as_index = False).size().sort_values(by = 'size', ascending = False).head(50))

    summary_ratings = summary_ratings[~summary_ratings['third_aggregation'].isin(['unrated'])]
    summary_ratings = summary_ratings.append({'third_aggregation': 'url_category_with_no_rating' , 'size': unrated_categorized}, ignore_index=True)
    summary_ratings = summary_ratings.append({'third_aggregation': 'no_category_no_rating' , 'size': unrated_uncategorized}, ignore_index=True)
    summary_ratings = summary_ratings.sort_values(by = 'size', ascending = False)

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

    cmap = plt.get_cmap(colormap)
    colors = [cmap(i) for i in np.linspace(0, 1, 10)]
    #colors = ['green', 'whitesmoke', 'limegreen', 'whitesmoke', 'orange', 'darkgreen', 'red', 'salmon', 'whitesmoke']

    wedges, texts = ax.pie(data, wedgeprops=dict(width=0.4), startangle=340, colors = colors)

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
    timestr = time.strftime("%Y_%m_%d")

    if type_df == 'rating':

        create_donut(x = 'third_aggregation',
                          y = 'size',
                          df = summary_ratings,
                          figure_name = 'summary_ratings_climate_{}_'.format(type) + timestr,
                          title = '',
                          type_title = type_title,
                          colormap = colormap)

    elif type_df == 'category':

        create_donut(x = 'category',
                          y = 'size',
                          df = summary_categories_unrated,
                          figure_name = 'summary_categories_climate_{}_'.format(type) + timestr,
                          title = '',
                          type_title = type_title,
                          colormap = colormap)

def create_figures(type_df):

    # create_donut_by_group(type = 'scientist',
    #                         type_title = 'Scientists',
    #                         colormap = 'Greens',
    #                         type_df = type_df)
    #
    # create_donut_by_group(type = 'activist',
    #                         type_title = 'Activists',
    #                         colormap = 'Oranges',
    #                         type_df = type_df)
    #
    # create_donut_by_group(type = 'delayer',
    #                         type_title = 'Delayers',
    #                         colormap = 'Reds',
    #                         type_df = type_df)

    create_donut_by_group(type = 'all',
                            type_title = 'All groups',
                            colormap = 'cool',
                            #colormap = ['green', 'lightgray', 'limegreen', 'limegreen', 'lightgray', 'bisque', 'darkgreen', 'red', 'salmon', 'lightgray'],
                            type_df = type_df)

def create_dataset():

    df = import_google_sheet ('domain_names_rating', 0)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    df1 = import_data('data.csv')
    df1 = df1.rename(columns = {'Domain': 'domain_name', 'MBFC factual': 'MBFC_factual'})
    df1 = df1[['domain_name','MBFC_factual' ]]

    df2 = import_data ('twitter_data_climate_tweets_2022_07_19_2.csv')

    df = df.dropna(subset = ['MBFC_factual'])


    print(len(df))
    print(len(df1))

    list_1 = df['domain_name'].tolist()
    list_2 = df1['domain_name'].tolist()

    list_manual = [x for x in list_1 if x not in list_2]
    df_rat = df[df['domain_name'].isin(list_manual)][['domain_name', 'category', 'MBFC_factual']]
    save_data(df_rat, 'ratings_MBFC_semi_manual.csv', 0)
    #print(list_manual)
    print(len(list_manual))
    print(df.columns)
    print(df1.columns)

if __name__ == '__main__':

    #create_figures(type_df = 'rating')
    #create_dataset()
    df, df_freq = get_domain_names_Twitter (type  = 'all')
    print('repeated 1', len(df_freq[df_freq['size'] == 1]))
    plot_hist(df = df_freq, var = 'size')
