
# Code for the project: twi_clim

## Twitter Users dataset: Scientists, Activists, Delayers

### Delayers

#### To get error report for the Desmog urls:

(mac) Give permission

```
chmod u+x ./code/report_urls_desmog.sh
```

Run the script:

```
 ./code/report_urls_desmog.sh
```

```
python3 ./code/create_twitter_users_lists.py
```

Get users metrics:

```
python3 ./code/collect_users_metrics.py
```


## Tweets dataset 

To collect the tweets of all users within each groups, from 2021-06-01 to 2021-12-01, run: 

```
python3 ./code/collect_twitter_data_climate.py
```

To collect Tweets containing the keyword COP26 during COP26, run:

 ```
python3 ./code/collect_twitter_data_COP26.py
```

## Topic Detection

 ```
python3 ./code/get_topics.py
```

## Citation and Network Analysis


```
python3 ./code/get_networks.py
```

```
python3 ./code/get_cocitation_network.py
```




