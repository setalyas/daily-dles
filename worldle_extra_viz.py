# -*- coding: utf-8 -*-
"""
Created on Sat Mar 15 14:21:39 2025

@author: setat
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.express as px
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns
import matplotlib.patches as mpatches

pd.DataFrame.iteritems = pd.DataFrame.items

colors = ['#009fbf', '#58508d']
people = ['Laura', 'Sami']

#%% Get data

worldle_countries = pd.read_csv('AmendedData\\worldle_countries.csv',
                                index_col=0,
                                keep_default_na=False)  # avoid Namibia=NA issue
worldle_scores = pd.read_csv('AmendedData\\worldle_results.csv',
                             index_col=0)

worldles = worldle_scores.merge(worldle_countries, how='left', left_on='date', right_index=True)
worldles.drop('id', inplace=True, axis=1)
worldles['date'] = pd.to_datetime(worldles['date'])

#%% Same country over time: any improvement?

worldles['iso2'].value_counts()
# Max repeats for a country is 14 (!)

repeats = worldles.pivot_table(values='score', index='iso2', columns=['person', 'date'])
# doesn't work -- includes each date, so is a sparse table of almost all NAs
# => need to rank them or something

repeats = worldles.copy()
repeats = repeats[['person', 'date', 'score', 'iso2', 'name']]
repeats.sort_values(['iso2', 'person', 'date'], inplace=True)
repeats.drop_duplicates(subset=['person', 'date'], keep='last', inplace=True)
for code in repeats['iso2'].unique():
    for p in people:
        person_code = (repeats['iso2']==code) & (repeats['person']==p)
        repeats.loc[person_code, 'date'] = repeats.loc[person_code, 'date'].rank()

repeatp = repeats.pivot_table(values='score', index='name', columns=['person', 'date'])

for p in people:
    repeatp.loc[:, (p, 'summary')] = None
    for i in repeatp.index:
        l = repeatp.loc[i, p].notna().sum()
        if l==0:
            cat = 'Other'
        elif l==1 and repeatp.loc[i, p].max()==1:
            cat = 'Nailed'
        else:
            cat = 'Other'
        if l>=2:
            first2 = repeatp.loc[i, (p, [1,2])].mean()
            last2 = repeatp.loc[i, (p, [l-1, l])].mean()
            nottop = repeatp.loc[i, (p, range(1,l+1))].dropna().nsmallest(l-1)
            # all (bar 1x 2) = 1
            if nottop.mean()==1 and repeatp.loc[i, p].max()<=2 and l>=2:
                cat = 'Nailed'
            # Monotonically decreasing
            elif repeatp.loc[i, p].dropna().is_monotonic_decreasing and l>0:
                cat = 'Solid learning'
            # first 2 aren't 1-2, last 2 are 1-2
            elif first2>=2 and last2<=2 and l>=4:
                cat = 'Messy learning'
            # Getting better at least
            elif last2<first2 and l>=4:
                cat = 'Messy improvement'
            # Getting worse
            elif last2>first2 and l>=4:
                cat = 'Worsening'
        repeatp.loc[i, (p, 'summary')] = cat

#%% Same country over time: viz

repbucket = repeatp.xs('summary', axis=1, level=1).value_counts().reset_index()
cats = {'Nailed': 0, 'Solid learning': 1, 'Messy learning': 2,
        'Messy improvement': 3, 'Worsening': 4, 'Other': 5}
repbucket['colour'] = repbucket['Laura'].replace(cats)
# repbucket['size'] = repbucket['count']

for p in people:
    repbucket[p] = repbucket[p].astype("category")
    repbucket[p] = repbucket[p].cat.set_categories(cats.keys(), ordered=True)
repbucket.sort_values(['Sami', 'Laura'], inplace=True)

# Parcats better than Sankey, which needs you to specifically order into 2 parallel cats
fig = go.Figure(go.Parcats(
    dimensions=[
        {'label': 'Sami', 'values': repbucket['Sami']},
        {'label': 'Laura', 'values': repbucket['Laura']},
    ],
    counts=repbucket['count'],
    bundlecolors=True
))
plot(fig)
# Impossible to colour, without colouring each line individually?? Nightmare!

#%% Same country over time: viz

repbucket_long = repeatp.xs('summary', axis=1, level=1)
for p in people:
    repbucket_long[p] = repbucket_long[p].astype("category")
    repbucket_long[p] = repbucket_long[p].cat.set_categories(cats.keys(), ordered=True)
    repbucket_long[f'{p}_colour'] = repbucket_long[p].replace(cats)
repbucket_long.sort_values(['Sami', 'Laura'], inplace=True)

for p in people:
    fig = px.parallel_categories(repbucket_long, dimensions=['Sami', 'Laura'],
                    color=f"{p}_colour", color_continuous_scale=px.colors.sequential.Inferno,
                    # labels={'sex':'Payer sex', 'smoker':'Smokers at the table', 'day':'Day of week'}
                    )
    # plot(fig)
    fig.write_html(f"Outputs\\worldle_samecountry_{p}.html")

# Learnings: Laura has nailed more, and has more 'learned' -- but we're both learning overall
# There are some that S has nailed that L hasn't (e.g. 5 to Other)

#%% Symm diffs of the categories per person

good_cats = ['Nailed', 'Solid learning', 'Messy learning', 'Messy improvement']
catl = len(good_cats)
for i in range(1,catl):
    so_far = list(cats.keys())[:i]
    print(f"====={so_far[-1]}=====")
    print('Laura has but Sami doesn\'t')
    print(repbucket_long[repbucket_long['Laura'].isin(so_far) & ~repbucket_long['Sami'].isin(so_far)].index)
    print('Sami has but Laura doesn\'t')
    print(repbucket_long[repbucket_long['Sami'].isin(so_far) & ~repbucket_long['Laura'].isin(so_far)].index)

# L has nailed loads, but hasn't got Norway, Japan, Yemen, Iceland, Mexico etc. down though S has

#%% Scores by continent

# country_names = worldles[['name', 'iso2', 'continent']].drop_duplicates()
# country_names['continent'] = country_names['continent'].replace({
#     'Insular Oceania': 'Oceania'})
# country_name_map = country_names.set_index('name').to_dict()
# name_iso2 = country_name_map['iso2']
# name_cont = country_name_map['continent']

worldcont = worldles.loc[:, ['person', 'date', 'score', 'continent']]
worldcont['year'] = worldcont['date'].dt.year
fig, axs = plt.subplots(2, 3, sharey=True, sharex=True, figsize=(13, 8))
ax_map = [('North America', (0,0)), ('Europe', (0,1)), ('Asia', (0,2)),
          ('South America', (1,0)), ('Africa', (1,1)), ('Oceania', (1,2))]
person_palette = dict(zip(people, colors[::-1]))
for c, a in ax_map:
    sns.violinplot(data=worldcont.loc[worldcont['continent']==c],
                   x='score', y='year', orient='h',
                   hue='person', hue_order=people, split=True, cut=0,
                   inner=None, ax=axs[a], palette=person_palette,
                   scale="width", scale_hue=False)
    axs[a].set_title(c)
    axs[a].set_xlabel('')
    if c != 'Europe':
        axs[a].get_legend().set_visible(False)
    else:
        axs[a].legend(bbox_to_anchor=(1, 0.75))
plt.suptitle('Worldle: guesses over time, by continent', fontsize=25)
plt.tight_layout()
plt.savefig('Outputs\\Worldle_continent_time.png')

#%% Scores by population / area (did anything buck the trend?)

# Tried to do based on the actual score, but differences were too subtle to read

country_info = worldle_countries.copy()
country_info = country_info.reset_index(drop=True)
country_info.drop_duplicates(inplace=True)

scorecattocol = {
    'Nailed': 'green',
    'Solid learning': 'blue',
    'Messy learning': 'purple',
    'Messy improvement': 'orange',
    'Worsening': 'red',
    'Other': 'grey'}

patch_list = []
for k, v in scorecattocol.items():
    patch_list.append(mpatches.Patch(color=v, label=k))

for p in people:
    score_cats = repeatp.loc[:, p]
    score_cats = score_cats.merge(country_info, how='left', left_index=True, right_on='name')
    score_cats['colour'] = score_cats['summary'].replace(scorecattocol)
    print(score_cats.head(1).T)
    fig, ax = plt.subplots(1, 1, figsize=(13, 10))
    for w, x, y, c in score_cats[['iso2', 'population', 'area', 'colour']].values:
        ax.plot((x,),(y,), 'o', c='r', alpha=0.01)
        ax.annotate(w, (x, y), ha='center', va='center', color=c)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Population (log scale)')
    ax.set_ylabel('Area (log scale)')
    ax.set_xlim(1000)
    ax.set_ylim(100)
    ax.set_title(f'Worldle: {p}\'s guesses over time, by area and population')
    ax.legend(handles=patch_list)
    plt.tight_layout()
    plt.savefig(f'Outputs\\Worldle_time_area_pop_{p}.png')

# Laura needs to learn Zimbabwe + Ecuador
# I need to remember Nigeria + Pakistan/Iran
