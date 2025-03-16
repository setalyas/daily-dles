# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 10:55:31 2025

@author: setat
"""

import pandas as pd
import requests
import time
import json

#%% Set variables

start_date = pd.Timestamp('2022-01-22')
end_date = pd.Timestamp('2025-03-13')
run_range = pd.date_range(start_date, end_date)

#%% Grab JSONs with country info

# https://teuteuf-dashboard-assets.pages.dev/data/worldle/games/2022/2022-01-22.json

pause_counter = 0
total_counter = 0

for date in run_range:
    # Make file path
    y = date.strftime('%Y')
    m = date.strftime('%m')
    d = date.strftime('%d')
    url = f'https://teuteuf-dashboard-assets.pages.dev/data/worldle/games/{y}/{y}-{m}-{d}.json'
    fp = f'RawData\worldle_results\{y}-{m}-{d}.json'
    # Grab file, save locally
    with open(fp, 'wb') as f:
        response = requests.get(url)
        f.write(response.content)
    pause_counter += 1
    total_counter += 1
    if pause_counter == 5:
        pause_counter = 0
        time.sleep(1)
        print(f'{total_counter} complete...')

#%% Sort into CSV

answers = {}

for date in run_range:
    y = date.strftime('%Y')
    m = date.strftime('%m')
    d = date.strftime('%d')
    fp = f'RawData\worldle_results\{y}-{m}-{d}.json'
    with open(fp, 'r') as f:
        dayfile = json.load(f)
    answers[date.strftime('%Y-%m-%d')] = dayfile['countryCode']

with open('AmendedData\\answers.json', 'w+') as f:
    json.dump(answers, fp=f)
