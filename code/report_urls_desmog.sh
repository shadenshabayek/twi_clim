#!/bin/bash

TODAY_DATE=$(date +"%Y-%m-%d")

minet resolve url_desmog "./data/url_desmog_climate.csv" > "./data/report_url_desmog.csv"
