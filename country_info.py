# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:22:56 2025

@author: setat
"""

import json
import pandas as pd

#%% Pull in data

with open('AmendedData\\answers.json', 'r') as f:
    answers_dict = json.load(f)

# https://teuteuf-dashboard-assets.pages.dev/data/common/countries.json

with open('RawData\countries.json', 'r', encoding='utf8') as f:
    countries_dict = json.load(f)

#%% Match to countries

countries = pd.DataFrame.from_dict(data=answers_dict, orient='index')
countries.columns = ['code']

country_info = pd.DataFrame(countries_dict)
country_info.drop('names', axis=1, inplace=True)

# countries.merge(country_info, how='inner', on='code')

#%% Get extra data

# # Get area data
# area_page = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_and_dependencies_by_area')
# country_area = area_page[1]
# country_area = country_area[['Country / dependency', 'Total in km2 (mi2)']]

# # Get pop + continent data

# pop_page = pd.read_html('https://en.wikipedia.org/wiki/List_of_countries_by_population_(United_Nations)')
# country_pop = pop_page[0]
# country_pop = country_pop[['Country or territory', 'Population (1 July 2023)',
#                            'UN continental region[1]',
#                            'UN statistical subregion[1]']]

# # Get ISO-3166 keys

# code_page = pd.read_html('https://en.wikipedia.org/wiki/ISO_3166-1_alpha-2')
# country_code = code_page[4]
# country_code = country_code[['Code', 'Country name (using title case)']]

# # Actually screw that, they don't have the ISO keys

#%% Get extra data with ISO key

country_deep = pd.read_csv('RawData\\countries.csv')
country_deep = country_deep[['country', 'iso2', 'population', 'area',
                             'continent', 'un_member']]
# Fix odd mis-codings
country_deep.loc[country_deep['country'].isin(['United Kingdom', 'Denmark']), 'un_member'] = True
country_deep.loc[country_deep['country']=='Namibia', 'iso2'] = 'NA'

#%% Merge

country_stats = country_info.merge(country_deep, how='left', left_on='code', right_on='iso2')

# What is missing data
print(country_stats.loc[country_stats['country'].isna(), ['code', 'name']])

# What has mismatching names, out of interest
mismatchname = country_stats[country_stats['name']!=country_stats['country']]
country_stats.drop(['country', 'code'], axis=1, inplace=True)

# Fill in Aland info (from https://en.wikipedia.org/wiki/%C3%85land)
country_stats.loc[country_stats['name']=='Åland', ['iso2', 'population', 'area', 'continent', 'un_member']] = ['AX', 30541, 1580, 'Europe', False]

country_stats.set_index('iso2', inplace=True)

# There are still a few missing bits
print(country_stats.isna().sum())  # population and area
country_stats.loc[country_stats['population'].isna() | country_stats['area'].isna(), 'name']
# Get stats from the wiki pages - permanent pop only
missing_stats = {
    'AQ': [0, 14200000],  # https://en.wikipedia.org/wiki/Antarctica
    'BV': [0, 49],  # https://en.wikipedia.org/wiki/Bouvet_Island
    'IO': [0, 54],  # https://en.wikipedia.org/wiki/British_Indian_Ocean_Territory
    'GF': [292354, 84000],  # https://en.wikipedia.org/wiki/French_Guiana
    'TF': [0, 439666],  # https://en.wikipedia.org/wiki/French_Southern_and_Antarctic_Lands
    'HM': [0, 368],  # https://en.wikipedia.org/wiki/Heard_Island_and_McDonald_Islands
    'MQ': [349925, 1128],  # https://en.wikipedia.org/wiki/Martinique
    'YT': [320901, 374],  # https://en.wikipedia.org/wiki/Mayotte
    'PS': [5483450, 6020],  # https://en.wikipedia.org/wiki/Palestine :(
    'RE': [896175, 2511],  # https://en.wikipedia.org/wiki/R%C3%A9union
    'GS': [0, 3903],  # https://en.wikipedia.org/wiki/South_Georgia_and_the_South_Sandwich_Islands
    'BQ': [30397, 322],  # https://en.wikipedia.org/wiki/Caribbean_Netherlands
    }
# fill in gaps
for iso, metrics in missing_stats.items():
    country_stats.loc[iso, ['population', 'area']] = metrics
# All gaps should be filled
assert country_stats.isna().all().all() == False

country_stats.to_csv('AmendedData\\country_stats.csv')

#%% Merge with Worldle answers by day

worldle_countries = pd.DataFrame.from_dict(answers_dict, orient='index')
worldle_countries.columns = ['iso2']
worldle_countries = worldle_countries.merge(country_stats, how='left',
                                            left_on='iso2', right_index=True)
worldle_countries.to_csv('AmendedData\\worldle_countries.csv')
