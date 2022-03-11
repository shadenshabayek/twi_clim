import os
import numpy as np

from datetime import date
from dotenv import load_dotenv
from time import sleep

from utils import (collect_twitter_data,
                    import_google_sheet,
                    import_data,
                    tic,
                    toc)

from create_twitter_users_lists import (get_list_desmog,
                                        get_list_scientists_who_do_climate,
                                        get_list_top_mentioned_by_type,
                                        get_list_dropped_top_mentioned,
                                        get_list_open_feedback,
                                        get_list_activists
                                        )


def get_all_lists():

    list_1 = get_list_desmog() + get_list_open_feedback()
    list_2 = get_list_scientists_who_do_climate()
    list_31, list_32, list_312 = get_list_top_mentioned_by_type()
    list_4 = get_list_activists()

    list_users = list_1 + list_2 + list_31 + list_32 + list_312 + list_4

    return list_users

def keep_three_groups(df):

    df1 = import_data('type_users_climate.csv')
    df1['type'] = df1['type'].astype(int)
    df_final = df.merge(df1, how = 'inner', on = ['username'])

    keep_type = [1, 2, 4]
    df_final = df_final[df_final['type'].isin(keep_type)]
    df_final['username'] = df_final['username'].str.lower()
    #print('Total number of tweets (type 1,2,4) before update:', len(df_final))

    return df_final

if __name__=="__main__":

    load_dotenv()

    df = import_data('twitter_data_climate.csv')
    df = keep_three_groups(df)

    df1 = import_data('twitter_data_climate_users_cop26.csv')
    df1 = keep_three_groups(df1)

    list_users = [x for x in df['username'].unique() if x not in df1['username'].unique()]
    print(list_users)
    # # l = len(df)
    # # print(l)
    # # print(df.iloc[l-1])
    # # print(df['created_at'].iloc[l-1])
    # df['username'] = df['username'].str.lower()
    # list_old = df['username'].unique().tolist()
    # list_users = get_all_lists()
    # #list_users = get_list_activists()
    # list_users = [x for x in list_users if x not in list_old]
    # print(list_users, 'length list remaining', len(list_users))
    list_users_tw =['from:' + user for user in list_users]

    tic()

    for query in list_users_tw :

        collect_twitter_data(
            query = query,
            start_time = '2020-08-17T23:00:00Z',
            end_time = '2021-08-17T23:00:05Z',
            #end_time = df['created_at'].iloc[l-1],
            bearer_token= os.getenv('TWITTER_TOKEN'),
            filename = os.path.join('.', 'data', 'twitter_data_climate'  + '.csv'),
            )
        sleep(3)
    toc()
