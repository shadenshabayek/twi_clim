#Code for the project: twi_clim

Different steps. 

## Twitter Users dataset: Scientists, Activists, Delayers

### Delayers

####To get error report for the Desmog urls:

(mac) Give permission

```
chmod u+x ./code/report_urls_desmog.sh
```

Run the script:

```
 ./code/report_urls_desmog.sh
```

## Tweets dataset 

To collect the tweets of all users within each groups, from 2021-06-01 to 2021-12-01, run: 

```
./code/collect_twitter_data_climate.py
```

To collect Tweets containing the keyword COP26 during COP26, run:

 ```
./code/collect_twitter_data_COP26.py
```




