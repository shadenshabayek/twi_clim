
import pandas as pd
import numpy as np
import os

from dotenv import load_dotenv
from utils import (get_user_metrics,
                    import_data,
                    import_google_sheet,
                    save_data)

import selenium

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

""" Desmog database, get list climate + twitter handles"""

def get_urls_desmog_list():

    df = import_data ('desmog_climate.csv')

    df = df.drop(columns=['Unnamed: 1'], axis=1)
    df['user'] = df['user'].astype(str)
    df['user'] = df['user'].str.lower()

    dict = {
    "\.": "",
    "(deceased)": "",
    "sir\-": "",
    "lord\-": "",
    " ": "-",
    "\’": "-",
    "\,": "",
    "\:": "",
    "è": "-",
    "é": "e",
    "ö": "o",
    "ó": "o",
    "ø": "o",
    "á": "a",
    "år": "a",
    "ü": "u",
    "\(": "",
    "\)": ""
     }

    df['user'] = df['user'].replace(dict, regex=True)
    df['user'] = df['user'].replace("patrick-jeff-condon", "jeff-condon", regex=True)
    df['user'] = df['user'].replace("hendrik-henk-tennekes", "hendrik-tennekes", regex=True)
    df['user'] = df['user'].replace("timothy-f-ball-tim-ball", "tim-ball", regex=True)
    df['user'] = df['user'].replace("the-thirteen-foundation", "thirteen-foundation", regex=True)
    df['user'] = df['user'].replace("william-a-dunn", "william-dunn", regex=True)
    df['user'] = df['user'].replace("hugh-w-ellsaesser", "hugh-ellsaesser", regex=True)
    df['user'] = df['user'].replace("a-alan-moghissi", "alan-moghissi", regex=True)

    list_users = df['user'].tolist()
    list =['https://www.desmog.com/' + user for user in list_users]

    df1 = pd.DataFrame(list, columns = ['url_desmog'])

    #save_data(df1, 'url_desmog_climate.csv', 0)
    return df1

def check_for_errors_urls_desmog():

    
def get_twitter_handles_desmog_climate():

    df = import_data('url_desmog_climate.csv')

    list_url = df['url_desmog'].tolist()

    list_handles = []
    list_urls = []

    for url in list_url:

        print(url)

        browser = webdriver.Chrome(ChromeDriverManager().install())
        browser.get(url)

        try:
            element = browser.find_element_by_xpath("//h2[@id='h-social-media']/following-sibling::ul")
            twitter_handles = re.findall("@(.*?) on Twitter", element.text)

        except:
            twitter_handles = []

        browser.quit()
        print(twitter_handles)

        list_urls.append(url)
        list_handles.append(twitter_handles)

    df1 = pd.DataFrame()

    df1['urls_desmog'] = list_urls
    df1['twitter_handle'] = list_handles

    #append = 1
    #append = 0

    #save_data(df1, 'tw_handles_climate.csv', append)

    return df1

def get_twitter_handles_desmog_openfeedback():

    df1 = import_data('tw_handles_climate.csv')
    df2 = import_data('openfeedback_users.csv')

def get_users_followers():

    load_dotenv()

    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                    list = list_twitter_handles_from_google_spreadsheet('tw_handles_climate') ,
                    filename = os.path.join('.', 'data', 'followers_twitter_desmog_climate'  + '.csv'))

def get_list_desmog():

    #df = import_google_sheet ('followers_twitter_desmog_climate')
    #if not linked to google spread sheet_instance
    df = import_data('followers_twitter_desmog_climate.csv')

    """drop politicians"""

    df = df.replace(r'^\s*$', np.nan, regex=True)
    df1 = df.dropna(subset=['category'])
    drop_politicians = df1['username'].tolist()
    df = df[~df['username'].isin(drop_politicians)]

    """ Take desmog users with 10k+ followers"""

    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    """Tony heller is in Desmog but with the twitter account with one underscore, now the second oen suspended"""

    list = df['username'].tolist() + ['tony__heller']

    print("The number of Twitter (desmog) handles is", len(list))

    return list

def collect_members_from_twitter_list():

    load_dotenv()
    filename = './data/memebers_twitter_list_scientists_who_do_science'  + '.csv'
    list_id = 1053067173961326594
    bearer_token = os.getenv('TWITTER_TOKEN')

    get_list_members(filename, list_id, bearer_token)

    df = import_data(filename)
    df = df.replace(r'^\s*$', np.nan, regex=True)

    return df

def get_list_scientists_who_do_climate():

    #df = import_google_sheet ('followers_twitter_list_scientists_who_do_science')
    #if not linked to google spread sheet_instance
    df = import_data('followers_twitter_list_scientists_who_do_science.csv')
    df = df.replace(r'^\s*$', np.nan, regex=True)

    """ Take users with 10k+ followers"""

    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    list = df['username'].tolist()
    #print(list)
    print("The number of Twitter (scientists who do science) handles is", len(list))

    return list

def get_list_open_feedback():

    #df = import_google_sheet ('followers_twitter_desmog_climate')
    df = import_data ('followers_twitter_desmog_climate.csv')
    df = df[df['source'] == 'EV']
    df = df[df['follower_count']>9999]
    df['username'] = df['username'].str.lower()

    list = df['username'].tolist()
    print('List Open Feddback:', len(list), 'users')

    return list

def get_list_activists():
    #source: https://www.earthday.org/19-youth-climate-activists-you-should-follow-on-social-media/
    #keep those with 10k followers
    list_manual = [
    'israhirsi',
    'xiuhtezcatl',
    'lillyspickup',
    'jamie_margolin',
    'nakabuyehildaf',
    'namugerwaleah',
    'anunade',
    'varshprakash',
    'jeromefosterii',
    'the_ecofeminist', #100_top_liked_tweets_cop26
    'sumakhelena', #100_top_liked_tweets_cop26
    'gretathunberg' #100_top_liked_tweets_cop26
    ]

    df = import_data('twitter_COP26.csv')
    df = df.dropna(subset = ['user_profile_description'])
    df['user_profile_description'] = df['user_profile_description'].str.lower()
    df['username'] = df['username'].str.lower()
    df = df[df['followers_count']>9999]

    list_keywords =['climate', 'activist']
    df1 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords))]
    list_users_cop = df1['username'].unique().tolist()

    list_keywords2 = ['environmentalist']
    df2 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords2))]
    list_users_cop2 = df2['username'].unique().tolist()

    list_keywords3 = ['environmental activist']
    df3 = df[df.user_profile_description.apply(lambda tweet: all(words in tweet for words in list_keywords3))]
    list_users_cop3 = df3['username'].unique().tolist()

    A = (set(list_manual) | set(list_users_cop) | set(list_users_cop2) | set(list_users_cop3))
    final_list = list(A)

    final_list.remove('bjornlomborg') #delayer, in the dataset of desmog!
    final_list.remove('kimjungil1984') #seems fake
    final_list.remove('dkfordicky') #pictures of himself
    final_list.remove('tonylavs') #other language
    final_list.remove('rkyte365') #conseil
    final_list.remove('lisabloom') #lawyer hors sujet ? mais suivi par extinction rebellion...
    final_list.remove('charlotte_cymru') #not always climate focused
    final_list.remove('jeanmanes') #embassador
    final_list.remove('ecosensenow') # in the dataset of desmog!
    final_list.remove('geoffreysupran') # among scientists who do climate
    final_list.remove('janet_rice') #pol

    #print(df[df['username'] == 'emmanuelfaber']['user_profile_description'])
    final_list.remove('emmanuelfaber') #he changed his discription: "ceo - climate and social business activist" +français

    print('cliamte activists:', len(final_list))

    return final_list

if __name__ == '__main__':
    #get_list_scientists_who_do_climate()
    #get_list_open_feedback()
    #et_list_desmog()

    #get_list_top_mentioned_by_type()

    #get_users_followers()
    #get_list_activists()
    get_twitter_handles_desmog_climate()
