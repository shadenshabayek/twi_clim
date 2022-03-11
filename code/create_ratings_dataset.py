import ast
import pandas as pd
import numpy as np

from pandas.api.types import CategoricalDtype
from ural import get_domain_name
from utils import (import_data,
                    import_google_sheet,
                    push_to_google_sheet,
                    save_data,
                    update_truncated_retweets)


"""create initial dataset to be imported to a google spreadsheet for human annotation"""

def get_domain_names_Twitter ():

    df1 = import_data ('twitter_data.csv')

    df2 = import_data ('twitter_data_climate.csv')
    df2 = update_truncated_retweets(df2, 'climate_retweets_full_length_2022_01_24.csv')

    df3 = import_data ('twitter_busyDrT_expanded_urls.csv')
    df3 = df3.iloc[1:]

    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    df = df.dropna(subset=['domain_name'])
    df = df[['username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'text']]

    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])

    df=df.explode('domain_name')

    a = ['twitter.com']
    df = df[~df['domain_name'].isin(a)]
    df = df.dropna(subset=['domain_name'])
    df['username'] = df['username'].str.lower()

    print('There are', df['domain_name'].nunique(), 'unique domain names, out of', df['domain_name'].count())

    return df

def get_MBFC_from_iffy():

    df = import_data('iffy_misinfo_domains_10_03_2021.csv')
    df = df.drop(['BF', 'FC', 'MBFC','PF', 'WI','Site Rank', '✓s', 'W', 'Name'], axis=1)
    df = df.rename(columns={"MBFC factual": "MBFC_factual", "Domain": "domain_name"})

    return df

def get_domains_with_no_rating (df, df_ratings, rating):

    #rating = 'MBFC_factual'

    df[rating]=''

    df.set_index('domain_name', inplace=True)
    df.update(df_ratings.set_index('domain_name'))
    df=df.reset_index()

    df[rating] = df[rating].replace('','unrated')
    df[rating] = df[rating].fillna('unrated')

    df_stat=df.groupby([rating], as_index=False).size()

    cat_order = CategoricalDtype([
                                "very-low",
                                "low",
                                "mixed",
                                "unrated",
                                "unrated_old",
                                "(satire)",
                                "mostly-factual",
                                "high",
                                "very-high"
                                ], ordered = True)

    df_stat[rating] = df_stat[rating].astype(cat_order)
    df_stat=df_stat.sort_values(rating)

    df[rating] = df[rating].astype(cat_order)
    df = df.sort_values(by=rating, ascending=True)

    find = df[df[rating] == 'unrated']
    find = find.groupby(['domain_name'], as_index = False).size().sort_values(by='size', ascending=False)
    find = find.rename(columns={"size": "repetition"})

    #save_data(find, 'unknown_ratings_twitter_domain_names.csv', 0)

    print(df_stat)
    print(find.head(10))

    return find

"""Human annotation

Step 1: open 'unknown_ratings_twitter_domain_names.csv'

Step 2: manually look on MBFC website for missing ratings that are the most repeated
save as 'missing_mbfc_ratings.csv'

examples:
channelnewsasia.com  was cited 69293 times and had a rating on the MBFC website
jpost.com was cited 47794 times and had a rating on the MBFC website
dailycaller.com was cited 25702 times and had a rating on the MBFC website

Step 3: manually create categories for top repeated unrated websites such as gov/platforms
save as 'categories.csv'

examples:
youtube.com was cited 11825 so we assigned it to the category platform
amazon.com was cited  1215 so we assigned it to the category commercial_books
"""

def get_ratings ():

    df1 = get_MBFC_from_iffy()
    df2 = import_data('missing_mbfc_ratings.csv')

    df = pd.concat([df1, df2],
                        axis=0,
                        ignore_index=True)

    df = df.sort_values(by='MBFC_factual', ascending=False)

    return df

def create_initial_ratings_categories_csv(rating = 'MBFC_factual'):

    df = get_domain_names_Twitter ()

    df_ratings = get_ratings ()
    df1 = get_domains_with_no_rating (df, df_ratings, rating)

    print(df1.head(10))
    print('there are', df1['domain_name'].nunique(), 'unique unrated domain names in the initial dataset')

    df2 = import_data("categories.csv")
    categories = df2['domain_name'].tolist()

    df3 = df1[~df1['domain_name'].isin(categories)]

    print(df3.sort_values(by='repetition', ascending=False).head(60))
    print('there are', df3['domain_name'].nunique(), 'remaining unique unretaed domain names')

    df_google = pd.concat([df_ratings, df2, df3], axis=0, ignore_index=True)
    #df_google['repetition'] = df_google['repetition'].astype('int')
    #save_data(df_google, "domain_names_rating.csv", 0)

    return df_google

"""update google spreadsheet if tweets including new websites are collected"""

#def update_google_sheet(filename, rating, sheet_title):
def update_google_sheet(filename, rating):

    df = get_domain_names_Twitter ()

    rating = 'aggregated_rating'

    df_google = import_google_sheet ('domain_names_rating')
    df_google = df_google.replace(r'^\s*$', np.nan, regex=True)
    df_google[rating] = df_google[rating].fillna('unrated_old')
    #print(df_google.head(20))

    df_ratings = df_google.dropna(subset=[rating])
    df_ratings = df_google[['domain_name', rating]]
    #print(df_ratings.head(20))

    df1 = get_domains_with_no_rating (df, df_ratings, rating)

    df_categories = df_google.dropna(subset=['category'])
    categories = df_categories['domain_name'].tolist()

    df2 = df1[~df1['domain_name'].isin(categories)]

    # push_to_google_sheet (filename = filename,
    #                        sheet_title = sheet_title,
    #                        sheet_number = 3,
    #                        df = df2,
    #                        new_worksheet = 1)

    print(df2.info())
    print(df2.head(60))
    return df2

"""safety checks"""
def double_check_ratings_google_sheet():

    df_googlesheet = import_google_sheet ('domain_names_rating')
    df_googlesheet = df_googlesheet.replace(r'^\s*$', np.nan, regex=True)
    df_googlesheet = df_googlesheet.dropna(subset=['MBFC_factual'])

    df = create_initial_ratings_categories_csv()
    df = df.dropna(subset=['MBFC_factual'])

    check = df.merge(df_googlesheet, how = 'inner', on = ['domain_name'])

    print(check['MBFC_factual_x'].equals(check['MBFC_factual_y']))

def check_misinformation_links_ratings():

    df_OF = import_data('unreliable_media.csv')
    df_OF['url'] = df_OF['url'].astype(str)

    for index, row in df_OF.iterrows():
        df_OF.at[index, 'domain_name']=get_domain_name(row['url'])

    df_OF = df_OF.drop(['url','appearancesCount'], axis=1)

    remove = ['facebook.com', 't.me', 'wordpress.com']
    df_OF = df_OF[~df_OF['domain_name'].isin(remove)]

    df_googlesheet = import_google_sheet ('domain_names_rating')
    df_googlesheet = df_googlesheet.replace(r'^\s*$', np.nan, regex=True)

    check = df_googlesheet.merge(df_OF, how = 'inner', on = ['domain_name'])
    check = check[['domain_name', 'repetition', 'aggregated_rating', 'name', 'misinformationAppearancesCount']].sort_values(by='aggregated_rating', ascending=False).reset_index()
    check = check.drop(['index'], axis=1)
    check=check.fillna('')

    push_to_google_sheet (filename = 'domain_names_rating',
                           sheet_title = 'open_feedback',
                           sheet_number = 2,
                           df = check,
                           new_worksheet = 1)


""" Check repitions only for climate"""


if __name__ == '__main__':

    #double_check_ratings_google_sheet()
    #check_misinformation_links_ratings()
    #update_google_sheet(filename = 'domain_names_rating', rating = 'aggregated_rating', sheet_title = 'new_domain_names')
    update_google_sheet(filename = 'domain_names_rating', rating = 'aggregated_rating')
