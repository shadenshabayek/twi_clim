#!/bin/bash

TODAY_DATE=$(date +"%Y_%m_%d")

minet resolve url_desmog "./data/url_desmog_climate_2022_03_11_2.csv" > "./data/report_url_desmog_${TODAY_DATE}.csv"
