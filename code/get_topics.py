from copy import deepcopy
from datetime import date
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy
from top2vec import Top2Vec

import pandas as pd
pd.options.mode.chained_assignment = None
import plotly.express as px
import matplotlib.colors as mcolors
import numpy as np
import re #pip install regex (python3.7!)
import time
import umap.plot
from plotly.colors import n_colors
import os


""" important note : open in an env with python 3.7 (anaconda) """

def import_data(file_name):

    data_path = os.path.join(".", "data", file_name)
    df = pd.read_csv(data_path, low_memory=False)
    return df

def save_data(df, file_name, append):

    file_path = os.path.join('.', 'data', file_name)

    if append == 1:
        df.to_csv(file_path, mode='a', header=False, index=False)
    else:
        df.to_csv(file_path, index=False)

    print(" '{}' is saved.".format(file_name))

def save_figure(figure_name):

    figure_path = os.path.join('.', 'figures', figure_name)
    plt.savefig(figure_path, bbox_inches='tight')
    print("The {} figure is saved.".format(figure_name))

def get_tweets_by_type():

    df  = import_data('twitter_data_climate_tweets_2022_03_15.csv')
    df = df[~df['query'].isin(['f'])]

    #list_scientists, list_activists, list_delayers, df_followers = get_lists_and_followers()
    df_users = import_data('climate_groups_followers.csv')
    list_scientists = df_users[df_users['source'].isin(['twitter_list_scientists_who_do_climate'])]['username'].tolist()
    list_activists = df_users[df_users['source'].isin(['cop26_hashtag'])]['username'].tolist()
    list_delayers = df_users[df_users['source'].isin(['desmog_climate_database', 'open_feedback'])]['username'].tolist()

    df['type'] = ''
    df['type'] = np.where(df['username'].isin(list_scientists), 'scientist', df['type'])
    df['type'] = np.where(df['username'].isin(list_activists), 'activist', df['type'])
    df['type'] = np.where(df['username'].isin(list_delayers), 'delayer', df['type'])

    print(df[~df['type'].isin(['scientist', 'activist','delayer'])]['username'].unique())
    print(df.groupby(['type'], as_index = False).size())
    return df

def get_url_free_text(text):

    text = re.sub(r'http\S+', '', text)
    return text

def get_mention_free_text(text):

    text = re.sub(r'@\S+', '', text)
    return text

def remove_covid_tweets(df):

    df['text'] = df['text'].str.lower()
    mylist = ['covid', 'mask', 'fauci', 'wuhan', 'vax', 'pandemic', 'corona', 'virus', 'pharmas']

    df1 = df[df.text.apply(lambda tweet: any(words in tweet for words in mylist))]
    list1 = df1['id'].tolist()
    print('Number of covid tweets', len(list1))

    df = df[~df['id'].isin(list1)]
    return df

def remove_tweets (df, remove_covid, set_topic_climate, remove_mentions):

    df['username'] = df['username'].str.lower()
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'] + '.'
    df = df[df['lang'] == 'en']
    df['text'] = df['text'].apply(get_url_free_text)

    if remove_covid == 1:
        df = remove_covid_tweets(df)
        print('total number of tweets all time, excluding covid tweets', len(df))

    if set_topic_climate == 1:
        list_1 = ['arctic',
                    'alarmist',
                    'antarctic',
                    'bleaching',
                    'carbon ',
                    'climate',
                    'CO2',
                    'emissions',
                    ' fire'
                    'forest',
                    'geological',
                    'greenhouse',
                    'glacier',
                    'glaciers',
                    'heatwave',
                    ' ice ',
                    'nuclear',
                    ' ocean ',
                    'oceans ',
                    'plant ',
                    'pollutant',
                    'pollution',
                    'polar',
                    'renewable',
                    'recycled',
                    'recycle',
                    'science',
                    'solar',
                    'species',
                    'warming',
                    'wildfire',
                    'wildfires',
                    'wind ',
                    'wildlife',
                    'weather']
        #list from topic 0
        list_2 = ['climate',
                    'scientists' ,
                    ' heat ',
                    'drought',
                    'environmental',
                    'nature',
                    'planet',
                    'warming ',
                    ' water ',
                    'ocean',
                    'heatwaves',
                    'emissions',
                    'adaptation',
                    'planet ' ,
                    'temperatures' ,
                    'ecosystems ',
                    'research',
                    'resilience',
                    'carbon',
                    'heatwave',
                    'fossil',
                     'fuel']

        mylist = list(set(list_1) | set(list_2))

        df = df[df.text.apply(lambda tweet: any(words in tweet for words in mylist))]
        print('there are', len(df), 'tweets that speak about climate')
        print('tweets about climate per group:', (df.groupby(['type'])['text'].count())/len(df))
        print('average number of tweets per group')

    if remove_mentions == 1:
        df['text'] = df['text'].apply(get_mention_free_text)

    return df

def get_documents(remove_covid, set_topic_climate, remove_mentions, cop26):

    timestr = time.strftime("%Y_%m_%d")

    df = get_tweets_by_type()
    df['date'] = pd.to_datetime(df['created_at']).dt.date
    print('total number of tweets', len(df))

    if cop26 == 1:
        df = df[(df['date']> date(2021, 10, 30)) & (df['date']<date(2021, 11, 13))]
        df = df.reset_index()
        print('total number of tweets COP26', len(df))

    print('total number of users', df['username'].nunique())

    df = remove_tweets (df, remove_covid, set_topic_climate, remove_mentions)

    df_tweets = pd.DataFrame(columns=['username',
                                      'text_tweets_concat',
                                      'type'])

    for user in df['username'].unique().tolist():

        df1 = df[df['username'] == user]
        type = df1['type'].iloc[0]
        document = ' '.join(df1['text'].tolist())
        print('length of total tweets of', user, 'is', len(document))

        df_tweets = df_tweets.append({
                    'username': user,
                    'text_tweets_concat': document,
                    'type': type},
                    ignore_index=True)

    print(df_tweets.groupby(['type']).size())

    if cop26 == 1:
        title = 'df_tweets_climate_3groups_concat_{}_'.format(cop26) + timestr + '.csv'
        save_data(df_tweets, title, 0)
    else :
        title = 'df_tweets_climate_3groups_concat_{}_'.format(cop26) + timestr + '.csv'
        save_data(df_tweets, title, 0)

    return df_tweets

def get_doc_top2vec(cop26):

    timestr = time.strftime("%Y_%m_%d")

    if cop26 == 0:

        title = 'df_tweets_climate_3groups_concat_{}_'.format(cop26) + timestr + '.csv'
        df = import_data(title)
        #df = get_documents(remove_covid = 1, set_topic_climate = 1, remove_mentions = 1, cop26 = 0)
        df['length']=df['text_tweets_concat'].apply(len)
        df=df[df['length']>140] #140*6

    elif cop26 == 1:
        title = 'df_tweets_climate_3groups_concat_{}_'.format(cop26) + timestr + '.csv'
        df = import_data(title)
        #df = get_documents(remove_covid = 1, set_topic_climate = 1, remove_mentions = 1, cop26 = 1)
        df['length']=df['text_tweets_concat'].apply(len)
        df = df[df['length']>140]

    #df['length'].hist(bins=100)
    #print('Number of remaining users after excluding very short docs', len(df))
    #save_figure('length_docs_tweets_topics_clim_3groups.jpg')

    return df

def bigrammer(doc):
    #sentence_stream = doc.split(" ")
    min_count=20
    cop26 = 1
    df = get_doc_top2vec(cop26)
    docs=df['text_tweets_concat'].values

    sentence_stream = [[x for x in doc.replace('\n',' ').split(" ") if len(x)>2] for doc in docs]

    bigram = Phrases(sentence_stream, min_count=min_count, threshold=min_count, delimiter=b' ')
    bigram_phraser = Phraser(bigram)
    sentence_stream = simple_preprocess(strip_tags(doc).replace('\n',' '), deacc=False)

    return bigram_phraser[sentence_stream]

def run_top2vec(cop26):

    df = get_doc_top2vec(cop26)

    """
    min_count (int (Optional, default 50)) – Ignores all words with total fre-
    quency lower than this. For smaller corpora a smaller min_count will be necessary.
    """

    min_count = 20

    model = Top2Vec(documents=df['text_tweets_concat'].values,
                    embedding_model='doc2vec',
                    speed='learn',
                    tokenizer=bigrammer,
                    min_count=min_count,
                    hdbscan_args={"min_cluster_size":5})#Shaden replaced 20 by 10 #Setting the min cluster size of HDBscan algorithm to 30 to avoid getting too many clusters

    timestr = time.strftime("%Y_%m_%d")
    title = './top2vec/climate_topics_{}_cluster_size_5_'.format(cop26) + timestr + '.mod'
    model.save(title)

def play_with_top2vec(model_title):

    model = Top2Vec.load(model_title)
    topic_sizes, topic_nums = model.get_topic_sizes()

    N=len(topic_sizes)

    print(N)
    print("topic_sizes", topic_sizes)
    print("topic_nums", topic_nums)

    color_map = plt.get_cmap("Set3")(np.linspace(0, 1, N))

    legende=[]
    legende_dict={}

    for i in range(N):
        legende.append(   ', '.join(model.get_topics()[0][i][:3]))
        legende_dict[i]=   ', '.join(model.get_topics()[0][i][:3])
        #legende.append('Topic ' + str(i) + ': ' + ', '.join(model.get_topics()[0][i][:3]))
        #legende_dict[i]='Topic ' + str(i) + ': ' + ', '.join(model.get_topics()[0][i][:3])

    print(legende_dict)

    plt.figure(figsize=(8,N/4))
    plt.barh(range(N), topic_sizes, color=color_map[::-1])#topic_sizestop_scores_per_document[doc]*top_scores_per_document[doc]*top_scores_per_document[doc]*top_scores_per_document[doc],label=(df.iloc[i]['Titre']),alpha=.5)
    plt.yticks(range(N),map(lambda x: legende_dict[x],range(N)))#,rotation=90)
    plt.legend()

    for i,top_words in enumerate(model.get_topics()[0]):
        print (i,' ,'.join(top_words[:50]),'\n')

    return N, color_map, legende_dict

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield ' '.join(l[i:i + n])

def br(strin):

    strinb=strin.split()
    chunks=divide_chunks(strinb, 12)
    #print (list(chunks))
    return '<br>'.join(list(chunks))#strin.replace('\n','<br>').replace('. ','.<br>')

def get_individual_topics(plot_topics, model_title, figure_title_topics, figure_title_users, cop26):

    timestr = time.strftime("%Y_%m_%d")
    model = Top2Vec.load(model_title)

    df = get_doc_top2vec(cop26)
    N, color_map, legende_dict = play_with_top2vec(model_title)

    umap_args = {
    "n_neighbors": 15,
    "n_components": 2, # 5 -> 2 for plotting
    "metric": "cosine"}

    #umap_model = umap.UMAP(**umap_args).fit(model._get_document_vectors(norm=False))
    #umap.plot.points(umap_model, labels=model.doc_top)
    #umap.plot.points(umap_model, cmap="rainbow", labels=np.array(list(map(lambda x: 'Topic ' + str(x) + ' ' +legende_dict[x],model.doc_top))))
    umap_model = umap.UMAP(**umap_args).fit(model._get_document_vectors(norm=False))

    df['text_tweets_concat'] = df['text_tweets_concat'].str[0:2500]
    df["Description_br"]=df["text_tweets_concat"].apply(br)

    df_px=pd.DataFrame({'x':umap_model.embedding_[:,0],'y':umap_model.embedding_[:,1],'username':df['username'].tolist(), 'type':df['type'].tolist(), 'Topic_Number':list(map(lambda x:str(x),model.doc_top)), 'Topic_Name':list(map(lambda x:'Topic ' + str(x) + ': '+ legende_dict[x],model.doc_top)), "Description":df['Description_br'].tolist()})

    title = 'dftop2vec_coordinates_{}_cluster_size_5_'.format(cop26) + timestr + '.csv'
    save_data(df_px, title, 0)

    if plot_topics == 1:

        cold=[]
        cold_map={}

        for i in range(N):
            cold_map["Topic "+ str(i)+': '+legende_dict[i]]=mcolors.rgb2hex(color_map[N-1-i])
            cold.append(mcolors.rgb2hex(color_map[N-1-i]))

        df_px ['dummy_column_for_size'] = 3.5
        symbols = ["8", "7", "17"]

        fig1=px.scatter(df_px,
                        x='x',
                        y='y',
                        size = 'dummy_column_for_size',
                        color='Topic_Name',
                        color_discrete_map=cold_map,
                        hover_name="Topic_Name",
                        hover_data=["username", "type", "Description"],
                        opacity=.7,
                        symbol="type",
                        symbol_sequence = symbols)#

        df_px ['dummy_column_for_size'] = 0.05

        df_px['type'] = df_px['type'].astype(str)
        symbols = ["0", "1", "2", "3", "5", "14", "17", "8", "6", "7", "8", "13"]
        fig2=px.scatter(df_px,
                        x='x',
                        y='y',
                        size = 'dummy_column_for_size',
                        color='type',
                        color_discrete_map={"scientist": "palegreen", "activist": "moccasin", "delayer": "lightcoral"},
                        hover_name="Topic_Name",
                        hover_data=["username", "type", "Description"],
                        opacity=.7,
                        symbol="Topic_Name",
                        symbol_sequence = symbols
                        )

        fig1.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )))

        fig1.write_html(figure_title_topics +  '.html', auto_open=True)

        fig2.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            font=dict(
                family="Courier New, monospace",
                size=12,
                color="#7f7f7f"
            )))

        fig2.write_html(figure_title_users  + '.html', auto_open=True)
        #fig2.write_html('/Users/shadenshabayek/Documents/Webclim/link_top2vec/top2vec_shaden.github.io/index.html', auto_open=True)
        #fig.show()
        #save_figure('global_embedding_climate.jpg')

def get_hierarchy_of_topics(model_title, dendro_title, model_scores_title, cop26):

    df = get_doc_top2vec(cop26)
    print(len(df))

    model = Top2Vec.load(model_title)
    N, color_map, legende_dict = play_with_top2vec(model_title)

    topic_nums, topic_score, topics_words, word_scores  = model.get_documents_topics(list(range(len(df))), reduced=False, num_topics=3)

    print(type(topic_nums))
    print(topic_nums.shape)

    df_test = pd.DataFrame(columns=['type'])
    df_test['type'] = df['type']
    df_test['username'] = df['username']
    if N > 2:
        df_test[['topic_nb_1','topic_nb_2','topic_nb_3']] = topic_nums
        df_test[['topic_score_1','topic_score_2','topic_score_3']] = topic_score
    else :
        df_test[['topic_nb_1','topic_nb_2']] = topic_nums
        df_test[['topic_score_1','topic_score_2']] = topic_score

    print(df_test.head(50))
    print(df_test[df_test['topic_score_1']>0.5])

    save_data(df_test, model_scores_title, 0)

    res = np.inner(model.topic_vectors,model.topic_vectors)
    res.shape

    Z = hierarchy.linkage(res, 'complete')

    plt.figure(figsize=(10,18))
    hd=hierarchy.dendrogram(Z, p=30,color_threshold=0.3*max(Z[:,2]),orientation='right',labels=list(map(lambda x: 'Topic ' + str(x) + ' ' +legende_dict[x],range(N))))
    plt.tight_layout()

    plt.savefig(dendro_title)

def plot_topic_prevelance_by_type(model_title, coord_title, figure_title):

    timestr = time.strftime("%Y_%m_%d")

    df = import_data(coord_title)
    df1 = df.groupby(['Topic_Number', 'type']).size()

    model = Top2Vec.load(model_title)
    topic_sizes, topic_nums = model.get_topic_sizes()

    N=len(topic_sizes)

    legende=[]
    legende_dict={}

    for i in range(N):
        legende.append( 'Topic ' + str(i) + ': ' +  ', '.join(model.get_topics()[0][i][:3]))
        legende_dict[i]=  'Topic ' + str(i) + ': ' + ', '.join(model.get_topics()[0][i][:3])
        #legende.append('Topic ' + str(i) + ': ' + ', '.join(model.get_topics()[0][i][:3]))
        #legende_dict[i]='Topic ' + str(i) + ': ' + ', '.join(model.get_topics()[0][i][:3])

    print(legende_dict)

    #plt.figure(figsize=(8,N/4))
    ax = df.groupby('type').Topic_Number.value_counts().unstack(0).plot.barh(color = ['moccasin', 'lightcoral', 'palegreen'])
    ax.set(ylabel=None)
    plt.yticks(range(N), map(lambda x: legende_dict[x],range(N)))#,rotation=90)
    #plt.legend([ 'Scientists', 'Activists', 'Delayers'])
    plt.legend()
    save_figure(figure_title)

    #df.groupby('type').Topic_Number.value_counts().unstack(0).plot.barh()
    print(df1)

def generate_word_cloud(model_title, figure_title, topic):

    model = Top2Vec.load(model_title)
    model.generate_topic_wordcloud(topic, "white")
    save_figure(figure_title)
    #plt.show()

def get_plots_stats(cop26):

    timestr = time.strftime("%Y_%m_%d")
    model_title = '/Users/shadenshabayek/Documents/Webclim/Twitter_Climate/top2vec/climate_topics_{}_cluster_size_5_'.format(cop26) + timestr + '.mod'

    N, color_map, legende_dict = play_with_top2vec(model_title = model_title)

    figure_title_topics = './top2vec/global_embedding_color_by_topic_{}_cluster_size_5_'.format(cop26) + timestr
    figure_title_users = './top2vec/global_embedding_color_by_user_{}_cluster_size_5_'.format(cop26) + timestr

    get_individual_topics(plot_topics = 1,
                            model_title = model_title,
                            figure_title_topics = figure_title_topics,
                            figure_title_users = figure_title_users,
                            cop26 = cop26)

    model_scores_title = 'scores_models_{}_cluster_size_5_'.format(cop26) + timestr + '.csv'
    dendro_title = 'dendro_{}_cluster_size_5_'.format(cop26) + timestr +'.pdf'

    get_hierarchy_of_topics(model_title = model_title,
                            dendro_title = dendro_title,
                            model_scores_title = model_scores_title,
                            cop26 = cop26)

    coord_title = 'dftop2vec_coordinates_{}_cluster_size_5_'.format(cop26) + timestr + '.csv'
    figure_title = 'topic_prevelance_by_type_{}_cluster_size_5_'.format(cop26) + timestr + '.jpg'

    plot_topic_prevelance_by_type(model_title = model_title,
                                coord_title = coord_title,
                                figure_title = figure_title)

    topic_0 = 0
    figure_title_0 = 'wordcloud_topic_{}_cluster_size_5_'.format(cop26) + str(topic_0) + '_' + timestr + '.jpg'
    generate_word_cloud(model_title = model_title, figure_title = figure_title_0 , topic = topic_0)

    topic_1 = 1
    figure_title_1 = 'wordcloud_topic_{}_cluster_size_5_'.format(cop26) + str(topic_1) + '_' + timestr + '.jpg'
    generate_word_cloud(model_title = model_title, figure_title = figure_title_1 , topic = topic_1)

    if N > 2 :
        topic_2 = 2
        figure_title_2 = 'wordcloud_topic_{}_cluster_size_5_'.format(cop26) + str(topic_2) + '_' + timestr + '.jpg'
        generate_word_cloud(model_title = model_title, figure_title = figure_title_2 , topic = topic_2)

if __name__=="__main__":

  print('hello')
  #get_documents(remove_covid = 1, set_topic_climate = 1, remove_mentions = 1, cop26 = 1)
  #run_top2vec(cop26 = 1)
  get_plots_stats(cop26 = 1)
