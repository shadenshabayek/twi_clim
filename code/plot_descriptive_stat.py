import ast
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import plotly.express as px
import time

from oauth2client.service_account import ServiceAccountCredentials
import gspread

# from create_twitter_users_lists import (get_list_desmog,
#                                         get_list_scientists_who_do_climate,
#                                         get_list_top_mentioned,
#                                         get_list_top_mentioned_by_type,
#                                         get_list_dropped_top_mentioned,
#                                         get_list_open_feedback,
#                                         get_list_activists)
from matplotlib import pyplot as plt

from utils import (import_data,
                    #import_google_sheet,
                    save_data,
                    save_figure
                    )

def keep_three_groups(df):

    df1 = import_data('type_users_climate.csv')
    df1['type'] = df1['type'].astype(int)
    df2 = df.merge(df1, how = 'inner', on = ['username'])

    keep_type = [1, 2, 4]
    df2 = df2[df2['type'].isin(keep_type)]

    print(df2.groupby(['type']).size())

    return df2

def get_info_description_climate():

    df1 = import_data('followers_twitter_list_scientists_who_do_science.csv')
    df2 = import_data('followers_twitter_desmog_climate.csv')
    df3 = import_data('followers_twitter_top_mentions_climate.csv')

    df = pd.concat([df1, df2, df3], axis=0, ignore_index=True)
    df = df[df['follower_count']>9999]

    list_1 = get_list_desmog()
    list_2 = get_list_scientists_who_do_climate()
    list_3 = get_list_top_mentioned()

    df['username'] = df['username'].str.lower()

    df['type'] = 3

    df['type'] = np.where(df['username'].isin(list_1), 1, df['type'])
    df['type'] = np.where(df['username'].isin(list_2), 2, df['type'])

    mylist = ['climate', 'warming']
    pattern = '|'.join(mylist)

    df = df.dropna(subset=['description'])
    df_clim = df[df['description'].str.contains(pattern)]

    print(df_clim[['username', 'follower_count', 'following_count', 'tweet_count', 'location', 'created_at','description', 'type']])
    print(df_clim.info())
    print(df_clim.groupby(['type']).size())
    return df_clim

def get_mentions_usernames_Twitter ():

    df = add_type_list_climate(df = import_data('twitter_data_climate.csv'))

    df = df.dropna(subset=['mentions_username'])

    for index, row in df.iterrows():
        df.at[index, 'mentions_username']=ast.literal_eval(row['mentions_username'])

    df = df.explode('mentions_username')

    df = df[['username', 'mentions_username', 'type_of_tweet', 'id', 'text', 'followers_count', 'type']]

    df = df.dropna(subset=['mentions_username'])

    df['mentions_username'] = df['mentions_username'].str.lower()
    df['username'] = df['username'].str.lower()

    return df

def get_cited_domain_names_Twitter ():

    df = add_type_list_climate(df = import_data('twitter_data_climate.csv'))

    df = update_truncated_retweets(df, 'climate_retweets_full_length_2022_01_24.csv')
    #these users don't have tweets over the whole start and end period, I think they got suspended temp.
    #nhc_atlantic shares no links
    drop_users = ['kencuccinelli',
                    'bigjoebastardi',
                    'lozzafox',
                    'americanthinker',
                    'nhc_atlantic']

    df = df[~df["username"].isin(drop_users)]
    df = df.dropna(subset=['domain_name'])

    df = df[['username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'text', 'followers_count', 'type']]

    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])

    df=df.explode('domain_name')

    a = ['twitter.com']

    df = df[~df['domain_name'].isin(a)]

    df = df.dropna(subset=['domain_name'])
    df['username'] = df['username'].str.lower()

    print('There are', df['domain_name'].nunique(), 'unique cited domain names')

    return df

"""STAT"""

def get_domains_categories ():

    df1 = import_google_sheet ('domain_names_rating')
    #print('number of unique domain names', df1['domain_name'].nunique())
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2 = get_cited_domain_names_Twitter ()
    df2['category']=''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2['category'] = df2['category'].replace('','uncategorized')
    #print(df2.groupby(['type', 'category'])['category'].size())

    remove = ['uncategorized']
    df2 = df2[~df2['category'].isin(remove)]

    return df2

def get_percentage_categories():

    df = get_domains_categories ()

    monetization_tools = ['commercial',
                        'commercial_books',
                        'crowdfunding_fundraising',
                        'event_organization_tool',
                        'social_media_tools_marketing_sharng_petitions']

    academic = ['academic',
                'scientific_journal']

    organizations = ['NGO',
                    'governmental',
                    'international_organization',
                    'local_organization']

    platforms = ['platform', 'podcasts']

    alternative_platforms = ['alternative_platform']


    df_categories = pd.DataFrame(columns=['username',
                                            'type',
                                            'monetization_tools',
                                            'academic',
                                            'organizations',
                                            'platforms',
                                            'alternative_platforms',
                                            'total_within_category'])

    print('total users with domains within a category', len(df['username'].unique()))

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]
        df_user = df[df['username'] == user ]
        total_urls = df_user['domain_name'].count()

        money = df_user[df_user['category'].isin(monetization_tools)]['category'].count()
        acad = df_user[df_user['category'].isin(academic)]['category'].count()
        orga = df_user[df_user['category'].isin(organizations)]['category'].count()
        plat = df_user[df_user['category'].isin(platforms)]['category'].count()
        alt = df_user[df_user['category'].isin(alternative_platforms)]['category'].count()

        share_monetization_tools = round((money / total_urls)*100, 2)
        share_academic = round((acad / total_urls)*100, 2)
        share_organizations = round((orga / total_urls)*100, 2)
        share_platforms = round((plat / total_urls)*100, 2)
        share_alternative_paltforms = round((alt / total_urls)*100, 2)

        df_categories = df_categories.append({
                    'username': user,
                    'type': type,
                    'monetization_tools': share_monetization_tools,
                    'academic' : share_academic,
                    'organizations': share_organizations,
                    'platforms': share_platforms,
                    'alternative_platforms': share_alternative_paltforms,
                    'total_within_category' : total_urls}, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_categories_' + timestr + '.csv'

    save_data(df_categories, title, 0)

    #save_data(df_categories, 'climate_percentage_categories.csv', 0)

    return df_categories

def get_domains_ratings (rating):

    df1 = import_google_sheet ('domain_names_rating')
    #print('number of unique domain names', df1['domain_name'].nunique())
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2 = get_cited_domain_names_Twitter ()

    #rating = 'MBFC_factual'
    df2[rating]=''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2[rating] = df2[rating].replace('','unrated')

    return df2

def get_percentage_rating (rating):

    #rating = 'MBFC_factual'
    df =  get_domains_ratings (rating)
    df_percentage_rating = pd.DataFrame(columns=['username',
                                                'type',
                                                'rating_negative',
                                                'rating_positive',
                                                'rating_mixed',
                                                'total_with_rating',
                                                'percentage_negative',
                                                'percentage_positive',
                                                'percentage_mixed'])

    remove = ['unrated', '(satire)']
    df = df[~df[rating].isin(remove)]

    positive = ['high', 'very-high', 'mostly-factual']
    negative = ['low', 'very-low']
    mixed = ['mixed']

    print('total users with rated domains', len(df['username'].unique()))

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]

        df_user = df[df['username'] == user ]

        total_with_rating = df_user[rating].count()

        rating_positive = df_user[df_user[rating].isin(positive)][rating].count()
        rating_negative = df_user[df_user[rating].isin(negative)][rating].count()
        rating_mixed = df_user[df_user[rating].isin(mixed)][rating].count()

        if total_with_rating > 0:

            per_neg = round((rating_negative / total_with_rating)*100, 2)
            per_pos = round((rating_positive / total_with_rating)*100, 2)
            per_mix = round((rating_mixed / total_with_rating)*100, 2)

        else:
            per_neg = 0
            per_pos = 0
            per_mix = 0

        df_percentage_rating = df_percentage_rating.append({
                    'username': user,
                    'type': type,
                    'rating_negative': rating_negative,
                    'rating_positive': rating_positive,
                    'rating_mixed': rating_mixed,
                    'total_with_rating': total_with_rating,
                    'percentage_negative': per_neg,
                    'percentage_positive': per_pos,
                    'percentage_mixed': per_mix}, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_rating_agg_' + timestr + '.csv'

    save_data(df_percentage_rating, title, 0)

    return df_percentage_rating

def get_percentage_unique_links (rating):

    #rating = 'MBFC_factual'

    df =  get_domains_ratings (rating)

    df_percentage_unique_links = pd.DataFrame(columns=['username',
                                                'type',
                                                'nb_unique_urls',
                                                'total_urls',
                                                'share_unique_url'
                                                ])

    #print('total users with rated domains', len(df['username'].unique()))

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]

        df_user = df[df['username'] == user ]

        nb_unique_urls = df_user['domain_name'].nunique()
        total_urls = df_user['domain_name'].count()

        share_unique_url = round((nb_unique_urls/total_urls)*100, 2)

        df_percentage_unique_links = df_percentage_unique_links.append({
                    'username': user,
                    'type': type,
                    'nb_unique_urls': nb_unique_urls,
                    'total_urls': total_urls,
                    'share_unique_url': share_unique_url
                }, ignore_index=True)

    #save_data(df_percentage_unique_links, 'climate_percentage_rating_agg.csv', 0)

    return df_percentage_unique_links

def get_percentage_general_stat ():

    df = add_type_list_climate(df = import_data('twitter_data_climate.csv'))
    df['type_of_tweet'] = df['type_of_tweet'].fillna('created_content')
    #print('total number of tweets', df['id'].nunique())
    #print('total number of users', df['username'].nunique())

    df1 = get_mentions_usernames_Twitter ()
    df2 = get_domains_ratings (rating = 'aggregated_rating')

    rating = 'aggregated_rating'

    # list1 = df[df['type'] == 1]['username'].tolist()
    # list2 = df[df['type'] == 2]['username'].tolist()
    # list3 = df[df['type'] == 31]['username'].tolist()
    # list4 = df[df['type'] == 32]['username'].tolist()
    # list5 = df[df['type'] == 312]['username'].tolist()

    df_percentage_general = pd.DataFrame(columns=['username',
                                                'type',
                                                'total_tweet_count',
                                                'share_retweets',
                                                'share_created_content',
                                                'share_in_reply_to',
                                                'mentions_count',
                                                'total_metions_count',
                                                'share_unique_mentions',
                                                'total_domain_count',
                                                'share_tweets_with_link',
                                                'share_tweets_with_mention',
                                                'share_tweets_with_rated_link'
                                                ])

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]

        df_user = df[df['username'] == user ]
        df_user_mentions = df1[df1['username'] == user]
        df_user_domains = df2[df2['username'] == user]

        remove = ['unrated']
        df_user_domains_rated = df_user_domains[~df_user_domains[rating].isin(remove)]
        total_domains_with_rating = df_user_domains_rated['domain_name'].count()

        total_tweet_count = df_user['id'].nunique()
        total_tweet_count_with_mention = df_user_mentions['id'].nunique()
        total_tweet_count_with_domain = df_user_domains['id'].nunique()

        total_mentions_count = df_user_mentions['mentions_username'].count()
        total_domain_count = df_user_domains['domain_name'].count()

        unique_mentions_count = df_user_mentions['mentions_username'].nunique()
        unique_domains_count = df_user_domains['domain_name'].nunique()

        if total_mentions_count > 0 :
            share_unique_mentions = round(100*(unique_mentions_count/total_mentions_count),2)
        else:
            share_unique_mentions = 0

        if total_tweet_count_with_domain > 0 :
            share_tweets_with_rated_link = round(100*(total_domains_with_rating/total_tweet_count_with_domain),2)
        else:
            share_tweets_with_rated_link = 0

        if total_tweet_count > 0 :

            retweet_count = df_user[df_user['type_of_tweet'] == 'retweeted']['type_of_tweet'].count()
            share_retweets = round(100*(retweet_count/total_tweet_count),2)

            created_content_count = df_user[df_user['type_of_tweet'] == 'created_content']['type_of_tweet'].count()
            share_created_content = round(100*(created_content_count/total_tweet_count),2)

            in_reply_to_count = df_user[df_user['type_of_tweet'] == 'replied_to']['type_of_tweet'].count()
            share_in_reply_to = round(100*(in_reply_to_count/total_tweet_count),2)

            mentions_count = round(100*(total_tweet_count_with_mention / total_tweet_count),2)

            share_tweets_with_link = round(100*(total_tweet_count_with_domain/total_tweet_count),2)
            share_tweets_with_mention = round(100*(total_tweet_count_with_mention/total_tweet_count),2)

        else :
            share_retweets = 0
            share_created_content = 0
            share_in_reply_to = 0
            mentions_count = 0
            share_tweets_with_link = 0
            share_tweets_with_mention = 0

        df_percentage_general = df_percentage_general.append({'username': user,
                                                                        'type': type,
                                                                        'total_tweet_count': total_tweet_count,
                                                                        'share_retweets': share_retweets ,
                                                                        'share_created_content': share_created_content,
                                                                        'share_in_reply_to': share_in_reply_to,
                                                                        'mentions_count' : mentions_count,
                                                                        'total_mentions_count': total_mentions_count,
                                                                        'share_unique_mentions':share_unique_mentions,
                                                                        'total_domain_count': total_domain_count,
                                                                        'share_tweets_with_link': share_tweets_with_link,
                                                                        'share_tweets_with_mention': share_tweets_with_mention,
                                                                        'share_tweets_with_rated_link': share_tweets_with_rated_link
                }, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_general_stat_' + timestr + '.csv'

    save_data(df_percentage_general, title, 0)

    return df_percentage_general

def get_engagement_metrics():

    df = add_type_list_climate(df = import_data('twitter_data_climate.csv'))
    #df['type_of_tweet'] = df['type_of_tweet'].fillna('created_content')
    df['total_engagement'] = df['like_count'] + df['retweet_count'] + df['reply_count']

    df_engagement = pd.DataFrame(columns=['username',
                                        'type',
                                        'mean_like_count',
                                        'mean_reply_count',
                                        'mean_retweet_count',
                                        'mean_total_engagement',
                                        'median_like_count',
                                        'median_reply_count',
                                        'median_retweet_count',
                                        'median_total_engagement',
                                        'followers_count'
                                        ])

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]
        followers_count = df[df['username'] == user ].followers_count.unique()[0]

        df_user = df[df['username'] == user ]


        mean_like_count = round(df_user['like_count'].mean())
        mean_reply_count = round(df_user['reply_count'].mean())
        mean_retweet_count = round(df_user['retweet_count'].mean())
        mean_total_engagement = round(df_user['total_engagement'].mean())

        median_like_count = round(df_user['like_count'].median())
        median_reply_count = round(df_user['reply_count'].median())
        median_retweet_count = round(df_user['retweet_count'].median())
        median_total_engagement = round(df_user['total_engagement'].median())

        df_engagement = df_engagement.append({'username': user,
                                            'type': type,
                                            'mean_like_count': mean_like_count,
                                            'mean_reply_count': mean_reply_count,
                                            'mean_retweet_count': mean_retweet_count,
                                            'mean_total_engagement': mean_total_engagement,
                                            'median_like_count': median_like_count,
                                            'median_reply_count': median_reply_count,
                                            'median_retweet_count': median_retweet_count,
                                            'median_total_engagement': median_total_engagement,
                                            'followers_count': followers_count },
                                            ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'engagement_general_stat_' + timestr + '.csv'
    save_data(df_engagement, title, 0)

    print('max like:', df_engagement['mean_like_count'].max())
    print('min like:', df_engagement['mean_like_count'].min())
    print('max reply:', df_engagement['mean_reply_count'].max())
    print('min reply:', df_engagement['mean_reply_count'].min())
    print('max retweet:', df_engagement['mean_retweet_count'].max())
    print('min retweet:', df_engagement['mean_retweet_count'].min())
    print('max total eng:', df_engagement['mean_total_engagement'].max())
    print('min total eng:', df_engagement['mean_total_engagement'].min())


    return df_engagement

def get_top_mentions ():

    df = get_mentions_usernames_Twitter ()

    types =[1, 2]
    df = df[df['type'].isin(types)]

    df_top = df.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(200)

    list1 = df[df['type'] == 1]['username'].tolist()
    list2 = df[df['type'] == 2]['username'].tolist()

    top1 = df_top[df_top['mentions_username'].isin(list1)]['mentions_username'].count()
    top2 = df_top[df_top['mentions_username'].isin(list2)]['mentions_username'].count()

    print(df_top.head(60))
    print('mentions in group 1', top1)
    print('mentions in group 2', top2)

    df_top = df_top.rename(columns={'mentions_username': 'username'})

    df_match = import_data('climate_percentage_rating_agg.csv')
    df_match = df_match[['username', 'type']]

    df_merge = df_top.merge(df_match, how = 'outer', on = ['username'])

    save_data(df_merge, 'climate_top_mentions.csv', 0)

    return df_merge

def get_top_mentions_by_type ():

    df = get_mentions_usernames_Twitter ()

    types =[1, 2]
    df = df[df['type'].isin(types)]

    df1 = df[df['type'] == 1]
    df2 = df[df['type'] == 2]

    df_top1 = df1.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(300)
    df_top2 = df2.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(300)

    df_top1['top_mentionned_by_list'] = 31
    df_top2['top_mentionned_by_list'] = 32

    df_top = pd.concat([df_top1, df_top2], axis=0, ignore_index=True)

    list_1 = get_list_desmog()
    list_2 = get_list_scientists_who_do_climate()
    list_3 = get_list_top_mentioned()

    df_top['type'] = 4

    df_top['type'] = np.where(df_top['mentions_username'].isin(list_1), 1, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_2), 2, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_3), 3, df_top['type'])

    df_top = df_top.rename(columns={'mentions_username': 'username'})

    other_lists = list_1 + list_2
    list = df_top['username'].tolist()
    print('nb of top mentions BEFORE removing existing users:', len(list))

    list = [x for x in list if x not in other_lists]
    print('nb of top mentions AFTER removing existing users:', len(list))

    list_drop = get_list_dropped_top_mentioned()

    repetitions = df.groupby(['mentions_username'], as_index=False).size().sort_values(by='size', ascending=False)
    repetitions = repetitions.rename(columns = {'size' : 'total_nb_mentions'})
    repetitions = repetitions[repetitions['total_nb_mentions']>10]
    user = repetitions['mentions_username'].tolist()

    #share = 100*(df_top[df_top['username'].isin(list)]['size'].sum()/ df_top['size'].sum())
    share = 100*(df_top[df_top['username'].isin(other_lists + list_drop)]['size'].sum()/ df[df['mentions_username'].isin(user)]['mentions_username'].count())
    print('share of existing users in mentions (including dropped politicians):', share)

    #list_drop = get_list_dropped_top_mentioned()

    #share_2 = 100*(df_top[df_top['username'].isin(list + list_drop)]['size'].sum()/ df_top['size'].sum())
    #print('share of existing users in mentions + dropped politicians (e.g. POTUS):', share_2)



    # df_match = import_data('followers_twitter_top_mentions_climate.csv')
    # df_match['username'] = df_match['username'].str.lower()
    # df_match = df_match[['username', 'follower_count']]
    #
    # df_merge = df_top.merge(df_match, how = 'outer', on = ['username'])
    # df_merge = df_merge[df_merge['follower_count']>9999]
    #
    # print(df_merge.head(100))
    #
    # save_data(df_merge, 'climate_top_mentions_by_type.csv', 0)
    #
    #
    #
    # return df_merge

def get_top_mentions_of_top_mentions_by_type ():

    df = get_mentions_usernames_Twitter ()

    types =[31, 32, 312]
    df = df[df['type'].isin(types)]

    df1 = df[df['type'] == 31]
    df2 = df[df['type'] == 32]
    df3 = df[df['type'] == 312]

    df_top1 = df1.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(300)
    df_top2 = df2.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(300)
    df_top3 = df3.groupby(['mentions_username'], as_index = False).size().sort_values(by='size', ascending=False).head(300)

    df_top1['top_mentionned_by_list'] = 431
    df_top2['top_mentionned_by_list'] = 432
    df_top3['top_mentionned_by_list'] = 4312

    #print(df_top1.info())
    #print(df_top2.info())
    #print(df_top3.info())
    df_top = pd.concat([df_top1, df_top2, df_top3], axis=0, ignore_index=True)

    list_1 = get_list_desmog()
    list_2 = get_list_scientists_who_do_climate()
    list_31, list_32, list_312 = get_list_top_mentioned_by_type()

    df_top['type'] = 4

    df_top['type'] = np.where(df_top['mentions_username'].isin(list_1), 1, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_2), 2, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_31), 31, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_32), 32, df_top['type'])
    df_top['type'] = np.where(df_top['mentions_username'].isin(list_312), 312, df_top['type'])

    #print(df_top.info())
    #print(df_top.head())
    #print(df_top.tail())

    #list1 = df[df['type'] == 1]['username'].tolist()
    #list2 = df[df['type'] == 2]['username'].tolist()


    # top1 = df_top[df_top['mentions_username'].isin(list1)]['mentions_username'].count()
    # top2 = df_top[df_top['mentions_username'].isin(list2)]['mentions_username'].count()
    #
    # print(df_top.head(60))
    # print('mentions in group 1', top1)
    # print('mentions in group 2', top2)

    df_top = df_top.rename(columns={'mentions_username': 'username'})
    #
    #df_match = import_data('followers_twitter_top_mentions_climate.csv')
    #df_match['username'] = df_match['username'].str.lower()
    #df_match = df_match[['username', 'follower_count']]

    #df_merge = df_top.merge(df_match, how = 'outer', on = ['username'])
    #df_merge = df_merge[df_merge['follower_count']>9999]

    #print(df_merge.head(100))
    print(df_top.head(70))

    list_1 = get_list_scientists_who_do_climate()
    list_2 = get_list_desmog()
    list_31, list_32, list_312 = get_list_top_mentioned_by_type()

    other_lists = list_1 + list_2 + list_31 + list_32 + list_312
    list = df_top['username'].tolist()
    print('nb of top mentions by top mentions BEFORE removing existing users:', len(list))

    list = [x for x in list if x not in other_lists]
    print('nb of top mentions by top mentions AFTER removing existing users:', len(list))

    list_drop = get_list_dropped_top_mentioned()
    #share = 100*(df_top[df_top['username'].isin(list)]['size'].sum()/ df_top['size'].sum())
    share = 100*(df_top[df_top['username'].isin(other_lists + list_drop)]['size'].sum()/ df['mentions_username'].count())
    print('share of existing users in mentions (including dropped politicians):', share)


    #
    # share_2 = 100*(df_top[df_top['username'].isin(list + list_drop)]['size'].sum()/ df_top['size'].sum())
    # print('share of existing users in mentions + dropped politicians (e.g. POTUS):', share_2)



    #save_data(df_top, 'climate_top_mentions_of_top_mentions_by_type.csv', 0)

    return df_top

def get_mentions_stat ():

    df = get_mentions_usernames_Twitter ()

    list1 = df[df['type'] == 1]['username'].tolist()
    list2 = df[df['type'] == 2]['username'].tolist()

    df_mentions = pd.DataFrame(columns=['group',
                                        'total_mentions',
                                        'total_unique_mentions',
                                        'unique_mentions_within_group',
                                        'total_mentions_within_group',
                                        'unique_mentions_accross_group',
                                        'total_mentions_accross_group'])

    df1 = df[df['type'] == 1]
    df2 = df[df['type'] == 2]

    total_mentions_1 =  df1['mentions_username'].count()
    total_unique_mentions_1 = df1['mentions_username'].nunique()
    unique_mentions_within_group_1 = df1[df1['mentions_username'].isin(list1)]['mentions_username'].nunique()
    total_mentions_within_group_1 = df1[df1['mentions_username'].isin(list1)]['mentions_username'].count()
    unique_mentions_accross_group_1 = df1[df1['mentions_username'].isin(list2)]['mentions_username'].nunique()
    total_mentions_accross_group_1 = df1[df1['mentions_username'].isin(list2)]['mentions_username'].count()

    df_mentions = df_mentions.append({
                'group': 'desmog',
                'total_mentions': total_mentions_1,
                'total_unique_mentions': total_unique_mentions_1,
                'unique_mentions_within_group': unique_mentions_within_group_1,
                'total_mentions_within_group':total_mentions_within_group_1,
                'unique_mentions_accross_group':unique_mentions_accross_group_1,
                'total_mentions_accross_group':total_mentions_accross_group_1
            }, ignore_index=True)

    total_mentions_2 =  df2['mentions_username'].count()
    total_unique_mentions_2 = df2['mentions_username'].nunique()
    unique_mentions_within_group_2 = df2[df2['mentions_username'].isin(list2)]['mentions_username'].nunique()
    total_mentions_within_group_2 = df2[df2['mentions_username'].isin(list2)]['mentions_username'].count()
    unique_mentions_accross_group_2 = df2[df2['mentions_username'].isin(list1)]['mentions_username'].nunique()
    total_mentions_accross_group_2 = df2[df2['mentions_username'].isin(list1)]['mentions_username'].count()

    df_mentions = df_mentions.append({
                'group': 'Scientists',
                'total_mentions': total_mentions_2,
                'total_unique_mentions': total_unique_mentions_2,
                'unique_mentions_within_group': unique_mentions_within_group_2,
                'total_mentions_within_group':total_mentions_within_group_2,
                'unique_mentions_accross_group':unique_mentions_accross_group_2,
                'total_mentions_accross_group':total_mentions_accross_group_2
            }, ignore_index=True)

    save_data(df_mentions, 'climate_mentions_groups.csv', 0)

    return df_mentions

"""SCORE"""

def get_score_cited_domains(df):

    df1 = import_data ('users_websites_expanded.csv')
    df1['domain_name'] = df1['domain_name'].fillna('no_website')

    df1['username'] = df1['username'].str.lower()
    df1 = df1.rename(columns = {"domain_name" : "own_website"})

    #compute sum of ratings of other websites with repetition:
    merged_df = df.merge(df1, how = 'outer', on = ['username'])
    merged_df = merged_df[merged_df['own_website'] != merged_df['domain_name']]

    merged_df['other_link_count_with_rating'] = 0

    for index, row in merged_df.iterrows():
        if row['rating'] < 0 :
            merged_df.at[index, 'other_link_count_with_rating']= 1
        elif row['rating'] > 0: #so if i cite websites with positie ratings it doesn't count
            #merged_df.at[index, 'rating']= 0
            merged_df.at[index, 'other_link_count_with_rating']= 1 #keep positives

    score = merged_df.groupby(['username', 'own_website', 'type_x'])[['rating', 'other_link_count_with_rating']].sum().reset_index()
    score = score.rename(columns={"rating": "sum_rating_cited_domains"})

    return score

def get_score_own_website(df, rating):

    df1 = import_data ('users_websites_expanded.csv')
    df1['domain_name'] = df1['domain_name'].fillna('no_website')
    df1['username'] = df1['username'].str.lower()
    df1 = df1.rename(columns = {"domain_name" : "own_website"})

    #compute sum of ratings of own website with repetitions:
    df1[rating]=''
    df1 = df1.rename(columns = {"own_website" : "domain_name"})

    df_ratings = import_google_sheet ('domain_names_rating')
    df_ratings = df_ratings[['domain_name', rating]]

    df1.set_index('domain_name', inplace=True)
    df1.update(df_ratings.set_index('domain_name'))
    df1 = df1.reset_index()
    df1[rating] = df1[rating].replace('','unrated')
    df1 = convert_rating_to_numbers(df1, rating)

    df1 = df1.rename(columns = {"domain_name" : "own_website"})
    df1 ['initial_score_MBFC'] = df1 ['rating']

    merged_df2 = df.merge(df1, how = 'inner', on = ['username'])
    merged_df2 = merged_df2[merged_df2['own_website'] == merged_df2['domain_name']]

    merged_df2['own_link_count_with_rating'] = 0

    for index, row in merged_df2.iterrows():
        if row['rating_y'] != 0 : #problem here
            merged_df2.at[index, 'own_link_count_with_rating']= 1

    score_2 = merged_df2.groupby(['username', 'own_website', 'type_x', 'initial_score_MBFC'], as_index=False)[['rating_y', 'own_link_count_with_rating']].sum().reset_index()
    score_2 = score_2.rename(columns={"rating_y": "sum_rating_own_website"})

    return score_2

def aggregate_cited_domains_and_own_website(score, score_2, df):

    final_df = score.merge(score_2, how = 'outer', on = ['username'])
    final_df = final_df.fillna(0)
    final_df = final_df.drop(['own_website_y', 'index'], axis=1)
    final_df = final_df.rename(columns={"own_website_x": "own_website"})
    final_df['own_link_count_with_rating'] = final_df['own_link_count_with_rating'].astype('int')

    final_df['total_nb_links'] = final_df['own_link_count_with_rating'] + final_df['other_link_count_with_rating']

    for index, row in final_df.iterrows():
        if row['total_nb_links']>0:
            final_df.at[index, 'final_score'] = (row['sum_rating_cited_domains'] + row['sum_rating_own_website'])/row['total_nb_links']
        else:
            final_df.at[index, 'final_score']=0

    final_df['final_score'] = round(final_df['final_score'],3)

    final_df['total_nb_links'] = final_df['total_nb_links'].astype('int')
    final_df['initial_score_MBFC'] = final_df['initial_score_MBFC'].astype('int')

    df2= df.drop_duplicates(subset=['username'], keep='first')
    df2 = df2 [['username', 'followers_count']]

    final_df = final_df.merge(df2, how = 'inner', on = ['username'])

    return final_df

def score(rating):

    df = get_domains_ratings (rating)

    df = convert_rating_to_numbers (df, rating)

    print(df.head(20))

    score = get_score_cited_domains (df)
    score_2 = get_score_own_website(df, rating)
    final_df = aggregate_cited_domains_and_own_website(score, score_2, df)

    #score_df = final_df.drop(['sum_rating_cited_domains', 'sum_rating_own_website', 'other_link_count_with_rating', 'own_link_count_with_rating'], axis=1)
    score_df = final_df.drop(['sum_rating_cited_domains', 'sum_rating_own_website', 'type_x_y'], axis=1)
    #score_df['final_score_round'] = round(score_df['final_score'])
    #score_df['final_score_round'] = score_df['final_score_round'].astype('int')

    score_df = score_df.rename(columns = {"followers_count" : "followers_count_twitter", "username": "Twitter_handle", "final_score":"final_score_twitter", "total_nb_links": "total_links_twitter", "total_nb_links": "total_nb_links_twitter", "type_x_x": "type"})

    print(score_df.sort_values(by = 'initial_score_MBFC', ascending = False).head(60))

    save_data(score_df, 'climate_score_postives_count_2.csv', 0)

    return score_df

"""PLOTS"""

def score_template(ax, median1, median2, m1, m2, m3, m4):

    #plt.legend(loc='upper right')
    plt.legend()
    #plt.axvline(x= median1 , color='red', linestyle='--', linewidth=1)
    plt.vlines(x= median1 , ymin=m1-0.05, ymax=m2+0.05, color='red', linestyle='--', linewidth=1)
    plt.text(median1+0.07, -0.38, "median", fontsize=7, color='red')
    #plt.text(median1+0.05, -0.4, "median", fontsize=7, color='red')

    #plt.axvline(x= median2 , color='green', linestyle='--', linewidth=1)
    plt.vlines(x= median2 , ymin=m3-0.05, ymax=m4+0.045, color='green', linestyle='--', linewidth=1)
    plt.text(median2+0.07, 0.765, "median", fontsize=7, color='green')
    #plt.text(median2+0.05, -0.4, "median", fontsize=7, color='green')

    plt.xticks([-3, -2, -1, 0, 1, 2, 3],
            ['-3', '-2', '-1', '0', '1', '2', '3'])
    plt.xlim(-3.5, 3.5)

    plt.yticks([])
    ax.set_frame_on(False)

#def percentage_rating_template(ax, median1, median2, m1, m2, m3, m4, median3, median4, m5, m6, m7, m8, median6, m9, m10, stat):
#def percentage_rating_template(ax, median1, median2, m1, m2, m3, m4, median3, median4, m5, m6, m7, m8, stat):
def percentage_rating_template(ax, median1, median2, m1, m2, m3, m4, median6, m9, m10, stat):

    #plt.legend(loc='upper right')
    plt.legend()
    #plt.axvline(x = 0, color='k', linestyle='--', linewidth=1)
    #axvline(linewidth=4, color='r')

    plt.vlines(x= median1 , ymin=m1, ymax=m2, color='red', linestyle='--', linewidth=1)
    plt.text(median1+0.33, m2, "median", fontsize=7, color='red')

    plt.vlines(x= median2 , ymin=m3, ymax=m4, color='green', linestyle='--', linewidth=1)
    plt.text(median2+0.33, m4, "median", fontsize=7, color='green')
    #plt.text(median2+0.3, 0.55, "median", fontsize=7, color='green')

    # plt.vlines(x= median3 , ymin=m5-0.06, ymax=m6+0.045, color='lightcoral', linestyle='--', linewidth=1)
    # plt.text(median3+0.3, 0, "median", fontsize=7, color='lightcoral')
    # #plt.text(median2+0.3, 0.55, "median", fontsize=7, color='green')
    #
    # plt.vlines(x= median4 , ymin=m7-0.05, ymax=m8+0.05, color='palegreen', linestyle='--', linewidth=1)
    # plt.text(median4+0.3, 0.2, "median", fontsize=7, color='palegreen')
    #plt.text(median2+0.3, 0.55, "median", fontsize=7, color='green')

    plt.vlines(x= median6 , ymin=m9, ymax=m10, color='orange', linestyle='--', linewidth=1)
    plt.text(median6+0.33, m10, "median", fontsize=7, color='orange')

    if stat == 1 :
        plt.xticks([0, 25, 50, 75, 100],
                ['0%', '25%', '50%', '75%', ' 100%'])
        plt.xlim(-1, 101)

    elif stat == 2 :

        #plt.xticks([0, 25, 50, 75, 100],
                #['0%', '25%', '50%', '75%', ' 100%'])
        #plt.xlim(0, 30000)
        plt.ticklabel_format(style = 'plain')


    plt.yticks([])
    ax.set_frame_on(False)

    #ax.spines['bottom'].set_position('zero')

def plot_bubbles(df, rating, xlabel, title, stat):

    #df = get_percentage_rating()
    #df = import_data ('climate_percentage_rating.csv')

    plt.figure(figsize=(5, 8))
    #plt.figure(figsize=(7, 5))
    #plt.figure(figsize=(7, 4))
    #plt.figure(figsize=(6, 3))
    ax = plt.subplot(111)



    random_y2 = list(np.random.random(len(df[df['type']==2]))/2+0.4)
    #random_y2 = list(np.random.random(len(df[df['type']==2]))/2+0.27)
    #random_y2 = list(np.random.random(len(df[df['type']==2]))/2)
    #print(np.random.random(len(df[df['type']==2])))

    plt.plot(df[df['type']==2][rating].values,
             random_y2,
             'o', markerfacecolor='green', markeredgecolor='green', alpha=0.6,
             #label='Scientists Who Do Climate')
             label='Scientists')


    # random_y3 = list(np.random.random(len(df[df['type']==31]))/4-0.03)
    # random_y4 = list(np.random.random(len(df[df['type']==32]))/4-0.03)
    # random_y5 = list(np.random.random(len(df[df['type']==312]))/4-0.03)
    #
    # plt.plot(df[df['type']==31][rating].values,
    #          random_y3,
    #          '*', markerfacecolor='lightcoral', markeredgecolor='lightcoral', alpha=0.6,
    #          label='Top mentioned by DCDD')
    #
    # plt.plot(df[df['type']==32][rating].values,
    #          random_y4,
    #          '*', markerfacecolor='palegreen', markeredgecolor='palegreen', alpha=0.6,
    #          label='Top mentioned by SWDC')
    #
    # plt.plot(df[df['type']==312][rating].values,
    #          random_y5,
    #          '*', markerfacecolor='deepskyblue', markeredgecolor='deepskyblue', alpha=0.6,
    #          label='Top mentioned common')

    # random_y6 = list(np.random.random(len(df[df['type']==4]))/2+0.9)
    #random_y6 = list(np.random.random(len(df[df['type']==4]))/5+0.79)
    random_y6 = list(np.random.random(len(df[df['type']==4]))/2-0.15)

    plt.plot(df[df['type']==4][rating].values,
             random_y6,
             '*', markerfacecolor='darkorange', markeredgecolor='orange', alpha=0.6,
             #label='Climate activists')
             label='Activists')

    #random_y1 = list(np.random.random(len(df[df['type']==1]))/4-0.32)
    random_y1 = list(np.random.random(len(df[df['type']==1]))/4-0.42)
    plt.plot(df[df['type']==1][rating].values,
             random_y1,
             'o', markerfacecolor='red', markeredgecolor='red', alpha=0.6,
             #label='Desmog Climate Disinfo')
             label='Delayers')

    median1 = np.median(df[df['type']== 1][rating])
    median2 = np.median(df[df['type']== 2][rating])

    # median3 = np.median(df[df['type']== 31][rating])
    # median4 = np.median(df[df['type']== 32][rating])

    median6 = np.median(df[df['type']== 4][rating])

    if stat == 1:
        percentage_rating_template(ax,
                                median1,
                                median2,
                                min(random_y1),
                                max(random_y1),
                                min(random_y2),
                                max(random_y2),
                                # median3,
                                # median4,
                                # min(random_y3),
                                # max(random_y3),
                                # min(random_y4),
                                # max(random_y4),
                                median6,
                                min(random_y6),
                                max(random_y6),
                                stat)
    elif stat == 2 :

        percentage_rating_template(ax,
                                median1,
                                median2,
                                min(random_y1),
                                max(random_y1),
                                min(random_y2),
                                max(random_y2),
                                median3,
                                median4,
                                min(random_y3),
                                max(random_y3),
                                min(random_y4),
                                max(random_y4),
                                median6,
                                min(random_y6),
                                max(random_y6),
                                stat)
    else:
        score_template(ax, median1, median2, min(random_y1), max(random_y1), min(random_y2), max(random_y2))

    plt.ylim(-.45, 1)
    plt.xlabel(xlabel, size='large')

    plt.tight_layout()
    save_figure(title)

def plot_share_ratings():

    timestr = time.strftime("%Y_%m_%d")
    #timestr = '2021_11_29' MBFC_reportingquality_scraped

    #plot_bubbles(df = get_percentage_rating(rating = 'aggregated_rating'),
    plot_bubbles(df = get_percentage_rating(rating = 'third_aggregation'),
                 #df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'percentage_negative',
                 xlabel = "Share of domains with low and very-low ratings " + timestr,
                 title = 'negative_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(#df = get_percentage_rating(rating = 'aggregated_rating'),
                 df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'percentage_positive',
                 xlabel = "Share of domains mostly-factual, high, very-high ratings " + timestr,
                 title = 'positive_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'percentage_mixed',
                 xlabel = "Share of domains with mixed rating " + timestr,
                 title = 'mix_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

def plot_terenary_graph():

    timestr = time.strftime("%Y_%m_%d")

    df = import_data('climate_percentage_rating_agg_' + timestr +'.csv')
    df = df[df['type'].isin([1,2,4])]
    df['type'] = df['type'].astype(str)
    #df['dummy_column_for_size'] = 0.000001

    fig = px.scatter_ternary(df,
                            a="percentage_positive",
                            b="percentage_negative",
                            c="percentage_mixed",
                            #size = 'dummy_column_for_size',
                            color='type',
                            color_discrete_map={"1": "red", "2": "green", "4": "orange"},
                            hover_name="username",
                            hover_data=["username", "rating_negative", "rating_positive", "rating_mixed", "total_with_rating"],
                            opacity=.6,)

    fig.update_traces(marker={'size': 3})
    fig.write_html('./figure/terenary_graph_ratings_' + timestr + '.html', auto_open=True)
    fig.write_html('/Users/shadenshabayek/Documents/Webclim/link_top2vec/terenary_graph_shaden.github.io/index.html', auto_open=True)
    fig.show()

def plot_share_categories():

    timestr = time.strftime("%Y_%m_%d")

    plot_bubbles(#df = import_data ('climate_percentage_categories.csv'),
                 df = get_percentage_categories(),
                 rating = 'monetization_tools',
                 xlabel = "share of links of monetization tools",
                 title = 'money_climate.jpg',
                 stat = 1 )

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_categories_' + timestr + '.csv'

    plot_bubbles(df = import_data (title),
                 rating = 'academic',
                 xlabel = "share of academic links",
                 title = 'academic_climate' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data (title),
                 rating = 'organizations',
                 xlabel = "share of links towards organizations (gov, international, local)",
                 title = 'organizations_climate' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data (title),
                 rating = 'platforms',
                 xlabel = "share of links towards other platforms (including podcasts)",
                 title = 'platforms_climate' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data (title),
                 rating = 'alternative_platforms',
                 xlabel = "share of links towards other alternative platforms",
                 title = 'alt_platforms_climate' + timestr + '.jpg',
                 stat = 1 )

def plot_share_general ():

    timestr = time.strftime("%Y_%m_%d")

    plot_bubbles(df = get_percentage_unique_links(rating = 'aggregated_rating'),
                 rating = 'share_unique_url',
                 xlabel = "share of unique URLS",
                 title = 'unique_urls_climate.jpg',
                 stat = 1)

    plot_bubbles(df = get_percentage_general_stat(),
                 #df = import_data (filename),
                 rating = 'share_unique_mentions',
                 xlabel = "Share of unique mentions",
                 title = 'unique_mentions_climate.jpg', stat = 1)

    filename = 'climate_percentage_general_stat_' + timestr + '.csv'

    plot_bubbles(df = import_data (filename),
                 rating = 'share_retweets',
                 xlabel = "Share of retweets",
                 title = 'retweets_climate.jpg',
                 stat = 1)

    plot_bubbles(df = import_data (filename),
                 rating = 'share_created_content',
                 xlabel = "Share of created content",
                 title = 'created_content_climate.jpg',
                 stat = 1)

    plot_bubbles(df = import_data (filename),
                 rating = 'share_in_reply_to',
                 xlabel = "Share of replies",
                 title = 'replies_climate.jpg',
                 stat = 1)

    plot_bubbles(df = import_data (filename),
                 rating = 'share_tweets_with_link',
                 xlabel = "Share of tweets containing a link",
                 title = 'links_climate.jpg',
                 stat = 1)

    plot_bubbles(df = import_data (filename),
                 rating = 'share_tweets_with_mention',
                 xlabel = "Share of tweets containing a mention",
                 title = 'mentions_climate.jpg',
                 stat = 1)


    plot_bubbles(df = import_data (filename),
                 rating = 'share_tweets_with_rated_link',
                 xlabel = "Share of tweets containing a rated link among tweets containing links " + timestr,
                 title = 'rated_links_climate_' + timestr + '.jpg',
                 stat = 1)

def plot_engagement_metric():

    timestr = time.strftime("%Y_%m_%d")
    #timestr = '2021_11_29'

    plot_bubbles(df = get_engagement_metrics(),
                 #df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'mean_like_count',
                 xlabel = "Mean like count of Tweets of each user " + timestr,
                 title = 'mean_like_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(#df = get_engagement_metrics(),
                 df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'mean_reply_count',
                 xlabel = "Mean reply count of Tweets of each user " + timestr,
                 title = 'mean_reply_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'mean_retweet_count',
                 xlabel = "Mean retweet count of Tweets of each user " + timestr,
                 title = 'mean_retweet_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'followers_count',
                 xlabel = "Followers count on Twitter of each user " + timestr,
                 title = 'followers_count_climate_' + timestr +'.jpg',
                 stat = 2 )

def plot_score ():

    # plot_bubbles(df = import_data('climate_score.csv'),
    #              rating = 'final_score_twitter',
    #              xlabel = "score based on domain citations (mixed=-1)",
    #              title = 'score_climate.jpg',
    #              stat = 0)

    # plot_bubbles(df = import_data('climate_score_mixed_0.csv'),
    #              rating = 'final_score_twitter',
    #              xlabel = "score based on domain citations (mixed=0)",
    #              title = 'score_climate_test_1.jpg',
    #              stat = 0)
    #
    timestr = time.strftime("%Y_%m_%d")

    plot_bubbles(df = import_data('climate_score_postives_count.csv'),
                 rating = 'final_score_twitter',
                 xlabel = "score based on domain citations, including positive ratings",
                 title = 'score_climate_test_2' + timestr + '.jpg',
                 stat = 0)

def plot_all():

    #plot_share_categories()
    plot_share_ratings()
    #plot_engagement_metric()
    #plot_share_general ()
    #plot_score()

def import_google_sheet (filename):

    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds = ServiceAccountCredentials.from_json_keyfile_name('./credentials.json', scope)
    client = gspread.authorize(creds)

    sheet = client.open(filename)
    sheet_instance = sheet.get_worksheet(0)

    records_data = sheet_instance.get_all_records()
    records_df = pd.DataFrame.from_dict(records_data)

    return records_df

if __name__ == '__main__':
    #score(rating = 'aggregated_rating')
    #plot_all()
    #plot_terenary_graph()
    df = import_google_sheet ('domain_names_rating')
    print(len(df))
    #add_type_list_climate(df = import_data('twitter_data_climate.csv'))
    #get_top_mentions_by_type ()
    #get_top_mentions_of_top_mentions_by_type ()
    #get_enagement_metrics()
    #get_top_mentions_by_type ()
    #get_top_mentions ()
    #get_info_description_climate()
    #add_type_list_climate(df = import_data('twitter_data_climate.csv'))
    #df = add_type_list_climate(df = import_data('twitter_data_climate.csv'))
