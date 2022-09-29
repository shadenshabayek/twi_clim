import csv
import time
import os

from dotenv import load_dotenv
from datetime import date
from time import sleep
from utils import (get_user_metrics,
                    import_data)

def get_active_accounts():

    with open('followers_twitter_delayers_climate.csv', 'r') as fin, open('followers_twitter_delayers_climate_active.csv', 'w', newline='') as fout:

        # define reader and writer objects
        reader = csv.reader(fin, skipinitialspace=True)
        writer = csv.writer(fout, delimiter=',')

        # write headers
        writer.writerow(next(reader))

        # iterate and write rows based on condition
        for i in reader:
            if i[-4] != "did not find the account, deleted or suspended":
                writer.writerow(i)

def get_inactive_accounts(filename):

    with open('followers_twitter_delayers_climate.csv', 'r') as fin, open(filename, 'w', newline='') as fout:

        # define reader and writer objects
        reader = csv.reader(fin, skipinitialspace=True)
        writer = csv.writer(fout, delimiter=',')

        # write headers
        writer.writerow(next(reader))

        # iterate and write rows based on condition
        for i in reader:
            if i[-4] == "did not find the account, deleted or suspended":
                writer.writerow(i)

def main():

    filename = 'followers_twitter_delayers_climate_inactive.csv'
    df = import_data(filename)
    list = df['username'].tolist()
    print(list)
    timestr = time.strftime("%Y_%m_%d")

    load_dotenv()
    get_user_metrics(bearer_token = os.getenv('TWITTER_TOKEN'),
                list = list,
                filename = os.path.join('.', 'data', 'user_metrics_inactive_' + timestr  + '.csv'),
                source = 'desmog_climate_database')

if __name__ == '__main__':
    
    main()
