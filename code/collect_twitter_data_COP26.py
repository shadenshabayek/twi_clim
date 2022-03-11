import os
import numpy as np

from dotenv import load_dotenv
from utils import collect_twitter_data

if __name__=="__main__":

    load_dotenv()

    collect_twitter_data(
        query = 'COP26 -is:retweet',
        start_time = '2021-10-31T01:00:00Z',
        end_time = '2021-11-12T23:00:05Z',
        bearer_token= os.getenv('TWITTER_TOKEN'),
        filename = os.path.join('.', 'data', 'twitter_COP26'  + '.csv'),
        )
