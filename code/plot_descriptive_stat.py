import ast
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import plotly.express as px
import time

#from oauth2client.service_account import ServiceAccountCredentials
#import gspread

from matplotlib import pyplot as plt

from utils import (import_data,
                    import_google_sheet,
                    save_data,
                    save_figure
                    )
from create_twitter_users_lists import get_lists_and_followers

def get_tweets_by_type():

    df  = import_data('twitter_data_climate_tweets_2022_07_19.csv')
    df['username'] = df['username'].str.lower()
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
    list_2 = ['johnredwood' 'climaterealists' 'tan123' 'netzerowatch' 'electroversenet',
    'marcelcrok' 'alexnewman_jou']

    list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()

    df['type'] = ''
    df['type'] = np.where(df['username'].isin(list_scientists), 'scientist', df['type'])
    df['type'] = np.where(df['username'].isin(list_activists), 'activist', df['type'])
    df['type'] = np.where(df['username'].isin(list_delayers), 'delayer', df['type'])

    df = df[df['type'].isin(['scientist', 'activist','delayer'])]

    return df

def get_cited_domain_names_Twitter () :

    df = get_tweets_by_type()

    df['positive_engagement'] = df['retweet_count'] + df['like_count']
    print(len(df))
    df = df[['username', 'domain_name', 'expanded_urls', 'type_of_tweet', 'id', 'text', 'followers_count', 'type', 'positive_engagement']]
    #df = df[~df['type_of_tweet'].isin(['replied_to'])]
    for index, row in df.iterrows():
        df.at[index, 'domain_name']=ast.literal_eval(row['domain_name'])
    #print('after removing replies', len(df))
    df = df.explode('domain_name')
    df = df.dropna(subset=['domain_name'])
    print('before removing Twitter links', len(df))

    a = ['twitter.com']

    df = df[~df['domain_name'].isin(a)]
    print('after removing Twitter links', len(df))

    df['username'] = df['username'].str.lower()
    df1 = df.groupby(['type'], as_index = False).size()
    print(df['domain_name'].head(30))
    print('There are', df['domain_name'].nunique(), 'unique cited domain names')
    print(df.groupby(['type'], as_index = False).size())
    print('unique tweets', df['id'].nunique())
    print('nb urls', len(df['expanded_urls']))

    return df, df1

def get_hashtags_by_type() :

    df = get_tweets_by_type()

    df = df[['username', 'hashtags', 'type_of_tweet', 'id', 'text', 'followers_count', 'type']]

    a = len(df[df['type'].isin(['activist'])])
    b = len(df[df['type'].isin(['delayer'])])
    c = len(df[df['type'].isin(['scientist'])])

    #df = df[~df['type_of_tweet'].isin(['replied_to'])]
    for index, row in df.iterrows():
        df.at[index, 'hashtags']=ast.literal_eval(row['hashtags'])

    df['nb_hashtags'] = df['hashtags'].apply(len)
    #print(df['nb_hashtags'].head(20))

    print('number of tw with hashtags', len(df[df['nb_hashtags']>0]))


    df = df.dropna(subset=['hashtags'])
    df = df[df['nb_hashtags']>0]
    print(df.head(40))
    df_count = df.groupby(['type'], as_index = False).size()
    print(df_count)
    print(df.groupby(['type'])['nb_hashtags'].agg('sum'))

    df = df.explode('hashtags')
    print('There are', df['hashtags'].nunique(), 'unique hastag')

    return df, df_count
"""STAT"""

def get_domains_categories ():

    df1 = import_google_sheet ('domain_names_rating', 0)
    #print('number of unique domain names', df1['domain_name'].nunique())
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2, df3 = get_cited_domain_names_Twitter ()
    df2['category']=''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2['category'] = df2['category'].replace('','uncategorized')
    #print(df2.groupby(['type', 'category'])['category'].size())

    #remove = ['uncategorized']
    #df2 = df2[~df2['category'].isin(remove)]

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

    df1 = import_google_sheet ('domain_names_rating', 0)
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2, df3 = get_cited_domain_names_Twitter ()

    df2[rating] = ''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2[rating] = df2[rating].replace('','unrated')

    return df2

def get_domains_bias (bias):

    df1 = import_google_sheet ('domain_names_rating', 1)
    df1 = df1.replace(r'^\s*$', np.nan, regex=True)

    df2, df3 = get_cited_domain_names_Twitter ()

    df2[bias] = ''

    df2.set_index('domain_name', inplace=True)
    df2.update(df1.set_index('domain_name'))
    df2=df2.reset_index()

    df2[bias] = df2[bias].replace('','unknown')

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
                                                'retweets_negative_rating',
                                                'retweets',
                                                'share_negative',
                                                'share_positive',
                                                'share_mixed',
                                                'positive_engagement_rating_negative',
                                                'positive_engagement_rating_positive',
                                                'positive_engagement_rating_mixed',
                                                'positive_engagement_all_links',
                                                'share_negative_weighted_engagement'])

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
        retweets = df_user[df_user['type_of_tweet'].isin(['retweeted'])][rating].count()

        rating_positive = df_user[df_user[rating].isin(positive)][rating].count()

        if rating_positive > 0 :
            positive_engagement_rating_positive = df_user[df_user[rating].isin(positive)]['positive_engagement'].mean()
            positive_engagement_rating_positive = round(positive_engagement_rating_positive)
        else :
            positive_engagement_rating_positive = 0

        rating_negative = df_user[df_user[rating].isin(negative)][rating].count()
        rn = df_user[df_user[rating].isin(negative)]
        # if user == 'drmarianeira':
        #     print(rn['id'])
        retweets_negative_rating = rn[rn['type_of_tweet'].isin(['retweeted'])][rating].count()

        if rating_negative > 0 :
            positive_engagement_rating_negative = df_user[df_user[rating].isin(negative)]['positive_engagement'].mean()
            positive_engagement_rating_negative = round(positive_engagement_rating_negative)
        else:
            positive_engagement_rating_negative = 0

        rating_mixed = df_user[df_user[rating].isin(mixed)][rating].count()

        if rating_mixed > 0:
            positive_engagement_rating_mixed = df_user[df_user[rating].isin(mixed)]['positive_engagement'].mean()
            positive_engagement_rating_mixed = round(positive_engagement_rating_mixed)
        else:
            positive_engagement_rating_mixed = 0

        positive_engagement_all_links = positive_engagement_rating_positive + positive_engagement_rating_negative + positive_engagement_rating_mixed

        if total_with_rating > 0:

            per_neg = round((rating_negative / total_with_rating)*100, 2)
            share_negative_weighted_engagement = round((positive_engagement_rating_negative)/(positive_engagement_all_links)*100,2)
            per_pos = round((rating_positive / total_with_rating)*100, 2)
            per_mix = round((rating_mixed / total_with_rating)*100, 2)

        else:
            per_neg = 0
            per_pos = 0
            per_mix = 0
            share_negative_weighted_engagement = 0

        df_percentage_rating = df_percentage_rating.append({
                    'username': user,
                    'type': type,
                    'rating_negative': rating_negative,
                    'rating_positive': rating_positive,
                    'rating_mixed': rating_mixed,
                    'total_with_rating': total_with_rating,
                    'retweets_negative_rating': retweets_negative_rating,
                    'retweets': retweets,
                    'share_negative': per_neg,
                    'share_positive': per_pos,
                    'share_mixed': per_mix,
                    'positive_engagement_rating_negative': positive_engagement_rating_negative,
                    'positive_engagement_rating_positive': positive_engagement_rating_positive,
                    'positive_engagement_rating_mixed': positive_engagement_rating_mixed,
                    'positive_engagement_all_links': positive_engagement_all_links,
                    'share_negative_weighted_engagement':share_negative_weighted_engagement}, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_rating_agg_' + timestr + '.csv'

    save_data(df_percentage_rating, title, 0)

    return df_percentage_rating

def get_percentage_bias (bias):

    #rating = 'MBFC_factual'
    df =  get_domains_bias (bias)

    df_percentage_bias = pd.DataFrame(columns=['username',
                                                'type',
                                                'total',
                                                'left',
                                                'right',
                                                'neutral',
                                                'questionable',
                                                'unknown',
                                                'share_left',
                                                'share_right',
                                                'share_neutral',
                                                'share_questionable',
                                                'share_unknown'])

    #remove = ['unrated', '(satire)']
    #df = df[~df[bias].isin(remove)]

    left_bias = ['Left-Center', 'Left']
    right_bias = ['Right-Center', 'Right']
    neutral_bias = ['Least Biased', 'Pro-Science']
    questionable_bias = ['Questionable Sources', 'Satire', 'Conspiracy-Pseudoscience']
    unknown_bias = ['unknown']

    print('total users with domains', len(df['username'].unique()))

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]

        df_user = df[df['username'] == user ]

        total_with_bias = df_user[bias].count()
        retweets = df_user[df_user['type_of_tweet'].isin(['retweeted'])][bias].count()

        left = df_user[df_user[bias].isin(left_bias)][bias].count()

        if left > 0 :
            share_left = left / total_with_bias
            share_left = round(share_left)
        else :
            share_let = 0

        right = df_user[df_user[bias].isin(right_bias)][bias].count()

        if right > 0 :
            share_right = right / total_with_bias
            share_right = round(share_right)
        else:
            share_right = 0

        neutral = df_user[df_user[bias].isin(neutral_bias)][bias].count()

        if neutral > 0:
            share_neutral = neutral / total_with_bias
            share_neutral = round(share_neutral)
        else:
            share_neutral = 0

        questionable = df_user[df_user[bias].isin(questionable_bias)][bias].count()

        if questionable > 0:
            share_questionable = questionable / total_with_bias
            share_questionable = round(share_questionable)
        else:
            share_questionable = 0


        unknown = df_user[df_user[bias].isin(unknown_bias)][bias].count()

        if unknown > 0:
            share_unknown = unknown / total_with_bias
            share_unknown = round(share_unknown)
        else:
            share_neutral = 0

        df_percentage_bias = df_percentage_bias.append({
                    'username': user,
                    'type': type,
                    'total': total_with_bias,
                    'left': left,
                    'right': right,
                    'neutral': neutral,
                    'questionable': questionable,
                    'unknown': unknown,
                    'share_left': share_left,
                    'share_right': share_right,
                    'share_neutral': share_neutral,
                    'share_questionable': share_questionable,
                    'share_unknown': share_unknown}, ignore_index=True)

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_bias_agg_' + timestr + '.csv'

    save_data(df_percentage_bias, title, 0)

    return df_percentage_bias

def get_percentage_unique_links (rating):

    df =  get_domains_ratings (rating)
    df_all = get_tweets_by_type()

    df_percentage_unique_links = pd.DataFrame(columns=['username',
                                                'type',
                                                'nb_unique_urls',
                                                'total_urls',
                                                'share_unique_url',
                                                'share_links_tweets'
                                                ])

    #print('total users with rated domains', len(df['username'].unique()))

    for user in df['username'].unique():

        type = df[df['username'] == user ].type.unique()[0]

        df_user_all_tweets = df_all  [df_all['username'] == user]

        df_user = df[df['username'] == user]

        nb_unique_urls = df_user['domain_name'].nunique()
        total_urls = df_user['domain_name'].count()

        share_links_tweets = round(100*(total_urls / len(df_user_all_tweets)), 2)

        share_unique_url = round((nb_unique_urls/total_urls)*100, 2)

        df_percentage_unique_links = df_percentage_unique_links.append({
                    'username': user,
                    'type': type,
                    'nb_unique_urls': nb_unique_urls,
                    'total_urls': total_urls,
                    'share_unique_url': share_unique_url,
                    'share_links_tweets': share_links_tweets
                }, ignore_index=True)

    save_data(df_percentage_unique_links, 'climate_percentage_rating_agg.csv', 0)

    return df_percentage_unique_links

def get_percentage_general_stat ():

    df = get_tweets_by_type()
    df['type_of_tweet'] = df['type_of_tweet'].fillna('created_content')
    #print('total number of tweets', df['id'].nunique())
    #print('total number of users', df['username'].nunique())

    df1 = get_mentions_usernames_Twitter ()
    df2 = get_domains_ratings (rating = 'aggregated_rating')

    rating = 'aggregated_rating'

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

    df = get_tweets_by_type()
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

    return df_engagement

def get_top_retweeted (type):

    df = get_tweets_by_type()
    type =[type]
    df = df[df['type'].isin(type)]

    df_top = df.groupby(['retweeted_username'], as_index = False).size().sort_values(by='size', ascending=False).head(30)
    print(df_top)

"""SCORE"""

def get_score_cited_domains(df):

    for index, row in merged_df.iterrows():
        if row['rating'] < 0 :
            merged_df.at[index, 'other_link_count_with_rating'] = 1
        elif row['rating'] > 0: #so if i cite websites with positie ratings it doesn't count
            #merged_df.at[index, 'rating']= 0
            merged_df.at[index, 'other_link_count_with_rating'] = 1 #keep positives

    score = merged_df.groupby(['username', 'own_website', 'type_x'])[['rating', 'other_link_count_with_rating']].sum().reset_index()
    score = score.rename(columns={"rating": "sum_rating_cited_domains"})

    return score

def score(rating):

    df = get_domains_ratings (rating)

    df = convert_rating_to_numbers (df, rating)

    print(df.head(20))

    score = get_score_cited_domains (df)

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

def percentage_rating_template(ax, median1, median2, median3, m1, m2, m3, m4, m5, m6, stat):

    #plt.legend(loc='upper right')
    plt.legend()

    plt.vlines(x= median1 , ymin=m1, ymax=m2, color='darkgreen', linestyle='--', linewidth=1)
    plt.text(median1+0.39, m2+0.015, "median", fontsize=7, color='green')

    plt.vlines(x= median2 , ymin=m3, ymax=m4, color='orange', linestyle='--', linewidth=1)
    plt.text(median2+0.33, m4+0.015, "median", fontsize=7, color='orange')

    plt.vlines(x= median3 , ymin=m5, ymax=m6, color='red', linestyle='--', linewidth=1)
    plt.text(median3+0.33, m6, "median", fontsize=7, color='red')

    if stat == 1 :
        plt.xticks([0, 25, 50, 75, 100],
                ['0%', '25%', '50%', '75%', ' 100%'])
        plt.xlim(-1, 101)

    elif stat == 2 :

        plt.ticklabel_format(style = 'plain')

    plt.yticks([])
    ax.set_frame_on(False)

    #ax.spines['bottom'].set_position('zero')

def plot_bubbles(df, rating, xlabel, title, stat):

    plt.figure(figsize=(5, 5))
    ax = plt.subplot(111)

    random_y1 = list(np.random.random(len(df[df['type'] == 'scientist']))/4+0.3)
    plt.plot(df[df['type']== 'scientist'][rating].values,
             random_y1,
             'o',
             markerfacecolor='limegreen',
             markeredgecolor='limegreen',
             alpha=0.6,
             label='Scientists')

    random_y2 = list(np.random.random(len(df[df['type']== 'activist']))/4)
    plt.plot(df[df['type']== 'activist' ][rating].values,
             random_y2,
             '*',
             markerfacecolor='darkorange',
             markeredgecolor='orange',
             alpha=0.6,
             label='Activists')

    random_y3 = list(np.random.random(len(df[df['type'] == 'delayer']))/4-0.3)
    plt.plot(df[df['type'] == 'delayer' ][rating].values,
             random_y3,
             'o',
             markerfacecolor='red',
             markeredgecolor='red',
             alpha=0.6,
             label='Delayers')

    median1 = np.median(df[df['type'] == 'scientist'][rating])
    median2 = np.median(df[df['type'] == 'activist'][rating])
    median3 = np.median(df[df['type'] == 'delayer'][rating])


    if stat == 1:
        percentage_rating_template(ax,
                                median1,
                                median2,
                                median3,
                                min(random_y1),
                                max(random_y1),
                                min(random_y2),
                                max(random_y2),
                                min(random_y3),
                                max(random_y3),
                                stat)
    elif stat == 2 :

        percentage_rating_template(ax,
                                median1,
                                median2,
                                median3,
                                min(random_y1),
                                max(random_y1),
                                min(random_y2),
                                max(random_y2),
                                min(random_y3),
                                max(random_y3),
                                stat)
    else:
        score_template(ax, median1, median2, min(random_y1), max(random_y1), min(random_y2), max(random_y2))

    plt.ylim(-0.4, 1)
    plt.xlabel(xlabel, size='large')

    plt.tight_layout()
    save_figure(title)

def plot_share_ratings():

    timestr = time.strftime("%Y_%m_%d")
    #timestr = '2021_11_29' MBFC_reportingquality_scraped

    plot_bubbles(df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
    #df = get_percentage_rating(rating = 'MBFC_factual'),
    #plot_bubbles(#df = get_percentage_rating(rating = 'third_aggregation'),
                 #df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'share_negative',
                 xlabel = "Share of domains rated\n low and very-low ",
                 title = 'negative_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(#df = get_percentage_rating(rating = 'aggregated_rating'),
                 df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'share_positive',
                 xlabel = "Share of domains rated \nmostly-factual, high, very-high ",
                 title = 'positive_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'share_mixed',
                 xlabel = "Share of domains rated as\n mixed ",
                 title = 'mix_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'share_negative_weighted_engagement',
                 xlabel = "Share of positive engagement (like & retweet count) \nfor Tweets with links rated as low or very-low",
                 title = 'weighted_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

    plot_bubbles(df = import_data ('climate_percentage_rating_agg_' + timestr +'.csv'),
                 rating = 'share_positive_weighted_engagement',
                 xlabel = "Share of positive engagement (like & retweet count) \nfor Tweets with links rated as low or very-low",
                 title = 'weighted_rating_climate_agg_' + timestr + '.jpg',
                 stat = 1 )

def plot_share_bias():

    bias = 'MBFC_bias_scraped'

    timestr = time.strftime("%Y_%m_%d")

    plot_bubbles(df = get_percentage_bias(bias = bias),
                 rating = 'share_left',
                 xlabel = "Share of domains with Center Left or Left bias ",
                 title = 'left_bias_climate_agg_' + timestr + '.jpg',
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
                 title = 'money_climate_' + timestr + '.jpg',
                 stat = 1 )

    timestr = time.strftime("%Y_%m_%d")
    title = 'climate_percentage_categories_' + timestr + '.csv'

    plot_bubbles(df = import_data (title),
                 rating = 'academic',
                 xlabel = "share of academic links",
                 title = 'academic_climate_' + timestr + '.jpg',
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
                 title = './stats/mean_like_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(#df = get_engagement_metrics(),
                 df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'mean_reply_count',
                 xlabel = "Mean reply count of Tweets of each user " + timestr,
                 title = './stats/mean_reply_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'mean_retweet_count',
                 xlabel = "Mean retweet count of Tweets of each user " + timestr,
                 title = './stats/mean_retweet_climate_' + timestr +'.jpg',
                 stat = 2 )

    plot_bubbles(df = import_data ('engagement_general_stat_' + timestr + '.csv'),
                 rating = 'followers_count',
                 xlabel = "Followers count on Twitter of each user " + timestr,
                 title = './stats/followers_count_climate_' + timestr +'.jpg',
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

def create_pie_figure(x, df, figure_name, title, labels, colors):

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_axes([0,0,1,1])
    ax.axis('equal')

    #labels= df[x].to_list()
    categories = df['size'].to_list()

    #cmap = plt.get_cmap('coolwarm')
    #colors = [cmap(i) for i in np.linspace(0, 1, len(labels))]

    patches, texts, pcts = ax.pie(
    categories,
    labels = labels,
    autopct='%.1f%%',
    wedgeprops={'linewidth': 3.0, 'edgecolor': 'white'},
    textprops={'fontsize': 29},
    colors=colors)

    plt.setp(pcts, color='white', fontweight='bold')
    save_figure(figure_name)

def plot_pies():

    #df, df1 = get_cited_domain_names_Twitter ()
    #df_tw = get_tweets_by_type()
    #df, df_count = get_hashtags_by_type()
    df_COP = import_data('twitter_COP26.csv')
    df_COP['type_of_tweet'] = df_COP['type_of_tweet'].fillna('raw_tweet')
    df_COP = df_COP.groupby(['type_of_tweet'], as_index= False).size()

    #df_tw['type_of_tweet'] = df_tw['type_of_tweet'].fillna('raw_tweet')
    #df2 = df_tw.groupby(['type_of_tweet'], as_index= False).size()
    #df3 = df_tw[df_tw['type'].isin(['scientist', 'activist', 'delayer'])].groupby(['type'], as_index= False).size()

    # x_1 = 'type'
    # a = df1[x_1].iloc[0]
    # b = df1[x_1].iloc[1]
    # c = df1[x_1].iloc[2]
    # prefixe = 'Climate '
    #
    # labels_1 = [ a + ' \n ({} links)'.format(df1['size'].iloc[0]),
    #            b + ' \n ({} links)'.format(df1['size'].iloc[1]),
    #             c + ' \n ({} links)'.format(df1['size'].iloc[2])]
    #
    # colors_1 = ['gold', 'lightcoral','lightgreen']
    #
    # create_pie_figure(x = x_1,
    #                   df = df1,
    #                   figure_name = 'share_url_clim.jpg',
    #                   title = '',
    #                   labels = labels_1,
    #                   colors = colors_1 )

    # create_pie_figure(x = 'type_of_tweet',
    #                   df = df2,
    #                   figure_name = 'share_type_tweet.jpg',
    #                   title = '',
    #                   labels = df2['type_of_tweet'].to_list(),
    #                   colors = ['plum', 'deepskyblue', 'lightgreen', 'pink'] )

    create_pie_figure(x = 'type_of_tweet',
                      df = df_COP,
                      figure_name = 'share_type_tweet_COP26.jpg',
                      title = '',
                      labels = df_COP['type_of_tweet'].to_list(),
                      colors = ['plum', 'deepskyblue', 'lightgreen', 'pink'] )

    # print(df3)
    # create_pie_figure(x = 'type',
    #                   df = df3,
    #                   figure_name = 'share_tweet_by_group.jpg',
    #                   title = '',
    #                   labels = df3['type'].to_list(),
    #                   colors = colors_1 )
    #
    # create_pie_figure(x = 'type',
    #                   df = df_count,
    #                   figure_name = 'share_hashtags_by_group.jpg',
    #                   title = '',
    #                   labels = df_count['type'].to_list(),
    #                   colors = colors_1 )

def plot_all():

    plot_pies()
    #plot_share_categories()
    plot_share_ratings()
    #plot_engagement_metric()
    #plot_share_general ()
    #plot_score()

# def import_google_sheet (filename):
#
#     scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
#     creds = ServiceAccountCredentials.from_json_keyfile_name('./credentials.json', scope)
#     client = gspread.authorize(creds)
#
#     sheet = client.open(filename)
#     sheet_instance = sheet.get_worksheet(0)
#
#     records_data = sheet_instance.get_all_records()
#     records_df = pd.DataFrame.from_dict(records_data)
#
#     return records_df

if __name__ == '__main__':


    #plot_share_ratings()
    #get_cited_domain_names_Twitter ()
    #plot_all()
    #get_hashtags_by_type()
    #get_tweets_by_type()
    plot_pies()
