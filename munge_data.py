# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 11:16:51 2024

@author: setat
"""

import re
import json
import pandas as pd
import matplotlib.pyplot as plt; plt.style.use('ggplot')
import matplotlib.ticker as mtick
import seaborn as sns

#%% Get data

files = ['RawData\\temp_phone.txt', 'RawData\\perm_phone.txt']
msg_files = []

for fp in files:
    with open(fp, 'r', encoding='utf-8') as f:
        msg_file = f.read()
    # print(msg_file)
    msg_files.append(msg_file)

print(msg_file[:1000])

#%% Extract messages

# Message start format = "27/07/2024, 07:00 - S: "
msg_heads = []
msg_bodies = []

for m in msg_files:
    heads = re.findall('\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - [^:]+: ', m)
    bodies = re.split('\d{2}/\d{2}/\d{4}, \d{2}:\d{2} - [^:]+: ', m)
    bodies = bodies[1:]  # remove the "Messages and calls are end-to-end..." row
    msg_heads += heads
    msg_bodies += bodies

#%% Other checks

# What happens to the 'security code' lines?
runs = 15
for h, b in zip(msg_heads[:runs], msg_bodies[:runs]):
    print('==========\n\n', h, b)
# Just added to the previous message -- fine, can ignore as there are only 2:
# re.findall('\n.+Tap to learn more.', msg_file)

# Check from the end
for h, b in zip(msg_heads[:-runs], msg_bodies[:-runs]):
    print('==========\n\n', h, b)

#%% Extract games

game_strs = {
    'Wordle': 'Wordle ([\d,]+)',
    'Worldle': '#Worldle #(\d+)',
    'travle': '#travle #(\d+)',
    'Waffle': '#waffle(\d+)',
    'Flagle': '#Flagle #(\d+)',
    'Framed': 'Framed #(\d+)',
    'Connections': 'Connections\s?\nPuzzle #(\d+)',
    'Octordle': 'Daily Octordle #(\d+)'
             }

game_messages = {k: {} for k in game_strs.keys()}

game_times = {'Sami': {}, 'Laura': {}}

for h, b in zip(msg_heads, msg_bodies):
    dt, person = re.split(' - ', h)
    date, time = re.split(', ', dt)
    if 'Laura' in person:
        person = 'Laura'
    elif '+44 7759 983951' in person:
        person = 'Laura'
    elif 'S:' in person:
        person = 'Sami'
    else:
        print(h, 'Error getting name')
    # print(date, person)
    for game, match in game_strs.items():
        if re.search(match, b):  # Needs to be search, as Laura often starts message with chat not results
            # print(game, '\n=====\n', b)
            game_id = re.search(match, b).group(1)
            if game_id not in game_messages[game]:
                game_messages[game][game_id] = {person: b,
                                               'date': date}
            else:
                game_messages[game][game_id][person] = b
            # Add times in
            time_i = pd.Timestamp(time) - pd.to_datetime('today').normalize()
            time_i = time_i.total_seconds() / (60*60)
            if date not in game_times[person]:
                game_times[person][date] = {game: time_i}
            else:
                game_times[person][date][game] = time_i

#%% Export to game_message JSON

with open('AmendedData\\game_messages.json', 'w', encoding='utf-8') as f:
    json.dump(game_messages, f, ensure_ascii=False, indent=4)

#%% Set up game *results*

game_results = {k: {} for k in game_strs.keys()}

#%% Framed results

framed_dict = {}

for i, d in game_messages['Framed'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for row in rows:
            if "🎥" in row:
                score_row = row
        if k != 'date':
            reds = score_row.count("🟥")
            if reds == 6:
                score = 7
            else:
                score = 1 + reds
            # print(i, k, score_row, score)
            if date not in framed_dict:
                framed_dict[date] = {k: score, 'id': i}
            else:
                framed_dict[date][k] = score

framed_results = pd.DataFrame.from_dict(framed_dict, orient='index')
framed_results.index = pd.to_datetime(framed_results.index, format='%d/%m/%Y')
framed_results.sort_index(inplace=True)

# Store
framed_results.to_csv('AmendedData\\framed_results.csv')
game_results['Framed'] = framed_results

#%% Wordle

wordle_dict = {}

for i, d in game_messages['Wordle'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for row in rows:
            # Wordle 1,003 3/6
            rgx = 'Wordle ([\d,]+) ([\dX])/6'
            if re.search(rgx, row):
                # print(row, re.search(rgx, row).group(2))
                raw_score = re.search(rgx, row).group(2)
        if k != 'date':
            if raw_score == 'X':
                score = 7
            else:
                score = int(raw_score)
            if date not in wordle_dict:
                wordle_dict[date] = {k: score, 'id': int(i.replace(',', ''))}
            else:
                wordle_dict[date][k] = score

wordle_results = pd.DataFrame.from_dict(wordle_dict, orient='index')
wordle_results.index = pd.to_datetime(wordle_results.index, format='%d/%m/%Y')
wordle_results.sort_index(inplace=True)

# Store
wordle_results.to_csv('AmendedData\\wordle_results.csv')
game_results['Wordle'] = wordle_results

#%% Waffle

waffle_dict = {}

for i, d in game_messages['Waffle'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for row in rows:
            # #waffle1031 4/5
            rgx = '#waffle(\d+) ([\dX])/5'
            if re.search(rgx, row):
                # print(row, re.search(rgx, row).group(2))
                raw_score = re.search(rgx, row).group(2)
        if k != 'date':
            if raw_score == 'X':
                score = -1 
            else:
                score = int(raw_score)
            if date not in waffle_dict:
                waffle_dict[date] = {k: score, 'id': int(i)}
            else:
                waffle_dict[date][k] = score

waffle_results = pd.DataFrame.from_dict(waffle_dict, orient='index')
waffle_results.index = pd.to_datetime(waffle_results.index, format='%d/%m/%Y')
waffle_results.sort_index(inplace=True)

# Store
waffle_results.to_csv('AmendedData\\waffle_results.csv')
game_results['Waffle'] = waffle_results

#%% Connections

connect_scores = []

"""🟪🟪🟪🟪
🟨🟩🟩🟩
🟩🟩🟩🟩
🟨🟨🟨🟨
🟦🟦🟦🟦"""

colours = ['yellow', 'green', 'blue', 'purple']

for i, d in game_messages['Connections'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for c, row in enumerate(rows):
            if 'Connections' in row:
                j_start = c
        if k != 'date':
            # skip the first 2 rows (ID rows), just get score rows
            score_rows = [i for i in rows[j_start+2:]]
            # Only if scores
            score_rows = [i for i in score_rows if re.match("[🟪🟨🟩🟦]", i)]
            colour_matches = {'yellow': {'emoji': '🟨'},
                              'green': {'emoji': '🟩'},
                              'blue': {'emoji': '🟦'},
                              'purple': {'emoji': '🟪'}
                              }
            for cs, d in colour_matches.items():
                try:
                    score = score_rows.index(d['emoji']*4) + 1
                except:
                    score = None
                colour_matches[cs]['score'] = score
            length = len(score_rows)
            results = [int(i), k, date]
            results += [colour_matches[colour]['score'] for colour in colours]
            connect_scores.append(results)

connect_results = pd.DataFrame(connect_scores,
                               columns=['id', 'person', 'date'] + colours)
connect_results['date'] = pd.to_datetime(connect_results['date'], format='%d/%m/%Y')
connect_results.sort_values('id', inplace=True)

# Store
connect_results.to_csv('AmendedData\\connect_results.csv')
game_results['Connections'] = connect_results

#%% Worldle

worldle_scores = []

for i, d in game_messages['Worldle'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for c, row in enumerate(rows):
            if '#Worldle' in row:
                score_row = c
            if 'https://worldle.teuteuf.fr' in row:
                rounds_row = c-1
        if k != 'date':
            total_score = rows[score_row]
            rgx = r'#Worldle #(\d+) (\([\d\.]+\) )?([\dX])/6'
            raw_score = re.search(rgx, rows[score_row]).group(3)
            if raw_score == 'X':
                score = 7
            else:
                score = int(raw_score)
            # 747 is around when the red flag emoji appears in scores
            if int(i) <= 747:
                sub_rounds = None
            else:
                sub_rounds = rows[rounds_row]
            # print(total_score, sub_rounds, raw_score)
            results = [int(i), k, date, score, sub_rounds]
            worldle_scores.append(results)
            sub_rounds

# Make into dataframe
worldle_results = pd.DataFrame(worldle_scores,
                               columns=['id', 'person', 'date', 'score', 'rounds'])
worldle_results['date'] = pd.to_datetime(worldle_results['date'], format='%d/%m/%Y')
worldle_results.sort_values('id', inplace=True)

# There is a hidden FE0F "variation selector" that messes up length/splitting
remove_fe0f = lambda c: re.sub(b'\xef\xb8\x8f', b"", c.encode("utf-8")).decode('utf-8')
assert len(remove_fe0f("🧭🚩🛡️🔤🗣️📐")) < len("🧭🚩🛡️🔤🗣️📐")
worldle_results['rounds'] = worldle_results.loc[worldle_results['rounds'].notna(), 'rounds'].apply(remove_fe0f)
worldle_results['round_score'] = worldle_results.loc[worldle_results['rounds'].notna(), 'rounds'].apply(len)

# One check per icon
icons = "🧭⭐📍🚩🛡🔤🗣👫🪙🏙📐"
icons = list(icons)
for i in icons:
    worldle_results[i] = worldle_results['rounds'].str.contains(i)

reduced_icons = icons.copy()
reduced_icons.remove('📍')
worldle_results['check_all'] = worldle_results[icons].all(axis=1)
worldle_results['check_most'] = worldle_results[reduced_icons].all(axis=1)
worldle_results['complete'] = ~(worldle_results['score'] == 7)

# Store
worldle_results.to_csv('AmendedData\\worldle_results.csv')
game_results['Worldle'] = worldle_results

#%% Flagle

flagle_dict = {}

for i, d in game_messages['Flagle'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for row in rows:
            # #Flagle #212 2/6
            # #Flagle #933 (11.09.2024) 2/6
            rgx = '#Flagle #(\d+) (\([\d\.]+\) )?([\dX])/6'
            if re.search(rgx, row):
                # print(row, re.search(rgx, row).group(3))
                raw_score = re.search(rgx, row).group(3)
        if k != 'date':
            if raw_score == 'X':
                score = 7
            else:
                score = int(raw_score)
            if date not in flagle_dict:
                flagle_dict[date] = {k: score, 'id': int(i)}
            else:
                flagle_dict[date][k] = score

flagle_results = pd.DataFrame.from_dict(flagle_dict, orient='index')
flagle_results.index = pd.to_datetime(flagle_results.index, format='%d/%m/%Y')
flagle_results.sort_index(inplace=True)

# Store
flagle_results.to_csv('AmendedData\\flagle_results.csv')
game_results['Flagle'] = flagle_results

#%% Octordle

# Daily Octordle #67
# 8️⃣9️⃣
# 5️⃣6️⃣
# 🔟7️⃣
# 🕚4️⃣
# octordle.com

# Daily Octordle #754
# 🔟🕛
# 9️⃣3️⃣
# 8️⃣7️⃣
# 6️⃣5️⃣
# Score: 60

valid_scores = {"1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7,
                "8": 8, "9": 9, "🔟": 10, "🕚": 11, "🕛": 12, "🕐": 13}
octordle_scores = []

for i, d in game_messages['Octordle'].items():
    date = d['date']
    for k, v in d.items():
        rows = v.split('\n')
        for c, row in enumerate(rows):
            if 'Daily Octordle' in row:
                j_start = c
        if k != 'date':
            # skip the first rows (ID row), just get score rows
            score_rows = [i for i in rows[j_start+1:]]
            score_rows = score_rows[:4]
            score_list = [list(i) for i in score_rows]
            score_list = [i for j in score_list for i in j]  # to one list
            score_list = [i for i in score_list if i in valid_scores.keys()]
            score_list = [valid_scores[i] for i in score_list]
            results = [int(i), k, date, min(score_list), max(score_list)]
            for s in sorted(score_list):
                results.append(s)
            octordle_scores.append(results)

octordle_results = pd.DataFrame(octordle_scores)
octordle_results.columns = ['id', 'person', 'date', 'min', 'max', '1', '2', '3',
                            '4', '5', '6', '7', '8']
octordle_results['date'] = pd.to_datetime(octordle_results['date'], format='%d/%m/%Y')
octordle_results.sort_values('id', inplace=True)

octordle_results['complete'] = ~octordle_results[['1', '2', '3', '4', '5', '6', '7', '8']].isna().any(axis=1)
octordle_results.loc[octordle_results['complete']==False, 'max'] = 14

# Store
octordle_results.to_csv('AmendedData\\octordle_results.csv')
game_results['Octordle'] = octordle_results

#%% Scoring plots: set-up

pct = mtick.PercentFormatter(xmax=1, decimals=0)
colors = ['#009fbf', '#58508d']
person = ['Sami', 'Laura']
games = list(game_strs.keys())

end = pd.Timestamp('2024-11-16')
start = end + pd.Timedelta(-30*6, 'days')

def make_scores(result_df):
    score_list = []
    for p in person:
        score_list.append(result_df[p].value_counts().sort_index())
    scores = pd.concat(score_list, axis=1)
    scores.columns = person
    scores = scores / scores.sum()
    return scores

def plot_bar(scores, name_str, xticklabels=[1, 2, 3, 4, 5, 6, 'X']):
    bar = scores.plot(kind='bar', color=colors, figsize=(8, 8))
    bar.set_title(f'{name_str}: complete on which guess')
    bar.set_ylabel('On guess')
    bar.set_xticklabels(xticklabels)
    bar.yaxis.set_major_formatter(pct)
    
def plot_cumu(scores, name_str, xmin=1, xmax=6):
    cumu = scores.cumsum().plot(color=colors, figsize=(8, 8), alpha=0.6)
    cumu.set_title(f'{name_str}: complete by which guess')
    cumu.set_xlabel('By guess')
    cumu.set_xlim(xmin, xmax)
    cumu.set_ylim(0, 1)
    cumu.yaxis.set_major_formatter(pct)


#%% Plot game time heatmap

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
for p, ax in zip(person, axs):
    time_list = [i for t in game_times[p].values() for i in t.values()]
    counts = pd.Series([int(t) for t in time_list]).value_counts()
    counts.sort_index(inplace=True)
    sns.heatmap(pd.DataFrame(counts), ax=ax, cmap='jet')
    ax.set_facecolor('#ffffff')
    ax.set_title(p, fontsize=20)
    ax.set_xlabel(None)
plt.suptitle("What time of day we did the 'dles", fontsize=35)
plt.tight_layout()
plt.savefig('Outputs\\total_timeofday.png')

#%% Plot times

# put into dfs, format dates
time_dfs = {}
for p in person:
    slc = pd.DataFrame(game_times[p]).T
    slc.index = pd.to_datetime(slc.index, dayfirst=True)
    slc['max'] = slc.max(axis=1)
    slc['min'] = slc.min(axis=1)
    time_dfs[p] = slc.copy().sort_index()

# Plot holistic spread
fig, axs = plt.subplots(1, 2, sharey=True, figsize=(21, 13))
for p, ax in zip(person, axs):
    sns.scatterplot(time_dfs[p][games], ax=ax, s=200, alpha=0.2,
                    hue_order=games)
    ax.invert_yaxis()
    ax.set_title(p, fontsize=20)
    ax.set_ylim(24, 0)
plt.suptitle("What time of day we did the 'dles", fontsize=35)
plt.tight_layout()
plt.savefig('Outputs\\spread_timeofday.png')

#%% Plot times: start vs finish

# plt.plot([8.6, 9.3], ['2024-05-20', '2024-05-20'], 'ro-')

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(21, 13))
for p, ax in zip(person, axs):
    slc = time_dfs[p].loc[start:end, ]
    for row in slc[['min', 'max']].iterrows():
        ax.plot(row[1].values, [row[0], row[0]], 'ko-', alpha=0.5)
    ax.set_title(p, fontsize=20)
    ax.set_xlim(0, 24)
axs[0].invert_yaxis()
plt.suptitle("Start to finish time gap", fontsize=35)
plt.tight_layout()
plt.savefig('Outputs\\startfinish_timeofday.png')

#%% Scoring: Worldle

score_range = range(1, 8)

worldle_scores = worldle_results[['person'] + icons].groupby('person').sum()
worldle_scores['rounds_count'] = worldle_results.loc[worldle_results['rounds'].notna(), ['person', 'score']].groupby('person').count()
worldle_scores['nonzero_score'] = worldle_results.loc[worldle_results['score']>0, ['person', 'score']].groupby('person').mean()
worldle_scores[['rounds_check_all', 'rounds_check_most']] = worldle_results.loc[worldle_results['rounds'].notna(), ['person', 'check_all', 'check_most']].groupby('person').sum()
# N.B. hard to assess, as not all iterations include all rounds
for i in icons:
    worldle_scores[i] = worldle_scores[i] / worldle_scores['rounds_count']
for name in ['Sami', 'Laura']:
    worldle_scores.loc[name, score_range] = worldle_results.loc[worldle_results['person']==name, 'score'].value_counts().sort_index().values
worldle_scores['participated'] = worldle_results[['person', 'score']].groupby('person').count()
for n in score_range:
    worldle_scores[n] = worldle_scores[n] / worldle_scores['participated']

# Scores: bar
wsc = worldle_scores[score_range].T.plot(kind='barh', color=colors, figsize=(8, 8))
wsc.invert_yaxis()
wsc.set_title('Worldle: guessing outlines')
wsc.set_ylabel('On guess')
wsc.set_yticklabels([1, 2, 3, 4, 5, 6, 'X'])
wsc.xaxis.set_major_formatter(pct)
plt.tight_layout()
plt.savefig('Outputs\\Worldle_guess.png')

# Scores: Cumu
wscc = worldle_scores[score_range].T.cumsum().plot(color=colors, figsize=(8, 8))
wscc.set_title('Worldle: guessing outlines')
wscc.set_xlabel('By guess')
wscc.yaxis.set_major_formatter(pct)
wscc.set_ylim(0, 1)
wscc.set_xlim(0, 5)
plt.tight_layout()
plt.savefig('Outputs\\Worldle_cumu.png')

# Scores: area
# wsc = worldle_scores[score_range].cumsum(axis=1).T.plot(kind='area', color=colors, stacked=False)
# wsc.yaxis.set_major_formatter(pct)
# wsc.set_xlim(0, 6)
# wsc.set_ylim(0, 1)
# wsc.set_title('Worldle: guessing outlines')
# wsc.set_xlabel('Guesses')
# wsc.set_ylabel('% guessed right by this guess')
# wsc.set_xticklabels([1, 2, 3, 4, 5, 6, 'X'])

# Worldle icons
icon_names = ['Neighbours', 'Capital', 'Pin', 'Flag', 'Emblem', 'Languages', 'Scripts',
              'Population', 'Currency', 'Cities', 'Area']
wic = worldle_scores[icons].T.plot(kind='bar', color=colors)
wic.yaxis.set_major_formatter(pct)
wic.set_xticklabels(icon_names)
wic.set_title('Worldle: follow-up rounds')
plt.tight_layout()
plt.savefig('Outputs\\Worldle_follow-ups.png')

#%% Over time: Worldle

fig, ax = plt.subplots(1, 1, figsize=(13, 8))
for p in person:
    slc = worldle_results.loc[worldle_results['person']==p, ['date', 'score']]
    slc = slc.sort_values('date')
    slc.set_index('date', inplace=True)
    slc.resample('14D').mean().plot(ax=ax)
ax.legend(person)
ax.set_xlabel('')
ax.set_title('Worldle: scores over time (rolling average)')
ax.set_ylim(0.9, 6)
plt.tight_layout()
plt.savefig('Outputs\\Worldle_overtime.png')

#%% Scoring: Connections

connect_results = game_results['Connections']

conn_c = ['yellow', 'green', 'blue', 'purple']

connect_results['first'] = connect_results[colours].idxmin(axis=1)
connect_results['mistakefree'] = (connect_results[colours].max(axis=1)==4) & (connect_results[colours].min(axis=1)==1)
connect_results['complete'] = ~connect_results[colours].isna().any(axis=1)
connect_results[colours] = connect_results[colours].fillna(8)

# Colour order
fig, axs = plt.subplots(4, 1, sharex=True, figsize=(5, 8))
for c, ax in zip(conn_c, axs):
    slc = connect_results[['person', c]].value_counts().unstack().copy()
    for n in slc.index:
        slc.loc[n, :] = slc.loc[n, :] / slc.sum(axis=1)[n]
    slc.T.plot(kind='bar', color=colors, ax=ax)
    ax.set_title(c.title())
    ax.yaxis.set_major_formatter(pct)
axs[3].set_xlabel('On guess')
axs[3].set_xticklabels([1, 2, 3, 4, 5, 6, 7, 'Missed'])
plt.suptitle('Connections: when each\ncolour was guessed right', fontsize=25)
plt.tight_layout()
plt.savefig('Outputs\\Connections_guess.png')

#%% Scoring: Flagle

# flagle_score_list = []
# for p in person:
#     flagle_score_list.append(flagle_results[p].value_counts().sort_index())
# flagle_scores = pd.concat(flagle_score_list, axis=1)
# flagle_scores.columns = person
# flagle_scores = flagle_scores / flagle_scores.sum()
flagle_scores = make_scores(flagle_results)

# Scores: bar
# fsc = flagle_scores.plot(kind='bar', color=colors, figsize=(8, 8))
# fsc.set_title('Flagle: correct on which guess')
# fsc.set_ylabel('On guess')
# fsc.set_xticklabels([1, 2, 3, 4, 5, 6, 'X'])
# fsc.yaxis.set_major_formatter(pct)
plot_bar(flagle_scores, name_str='Flagle')
plt.tight_layout()
plt.savefig('Outputs\\Flagle_guess.png')

# Scores: Cumu
# fscc = flagle_scores.cumsum().plot(color=colors, figsize=(8, 8), alpha=0.6)
# fscc.set_title('Flagle: correct by which guess')
# fscc.set_xlabel('By guess')
# fscc.set_xlim(1, 6)
# fscc.set_ylim(0, 1)
# fscc.yaxis.set_major_formatter(pct)
plot_cumu(flagle_scores, name_str='Flagle')
plt.savefig('Outputs\\Flagle_cumu.png')

#%% Over time: Flagle

fig, ax = plt.subplots(1, 1, figsize=(13, 8))
flagle_results[['Sami', 'Laura']].resample('14D').mean().plot(ax=ax)
ax.set_title('Flagle: scores over time (rolling average)')
ax.set_ylim(1, 6)
plt.tight_layout()
plt.savefig('Outputs\\Flagle_overtime.png')

#%% Scoring: Framed

framed_scores = make_scores(framed_results)

plot_bar(framed_scores, name_str='Framed')
plt.tight_layout()
plt.savefig('Outputs\\Framed_guess.png')

plot_cumu(framed_scores, name_str='Framed')
plt.tight_layout()
plt.savefig('Outputs\\Framed_cumu.png')

# framed_score_list = []
# for p in person:
#     framed_score_list.append(framed_results[p].value_counts().sort_index())
# framed_scores = pd.concat(framed_score_list, axis=1)
# framed_scores.columns = person
# framed_scores = framed_scores / framed_scores.sum()

# # Scores: bar
# frsc = framed_scores.plot(kind='bar', color=colors, figsize=(8, 8))
# frsc.set_title('Framed: correct on which guess')
# frsc.set_ylabel('On guess')
# frsc.set_xticklabels([1, 2, 3, 4, 5, 6, 'X'])
# frsc.yaxis.set_major_formatter(pct)

# # Scores: Cumu
# frscc = framed_scores.cumsum().plot(color=colors, figsize=(8, 8), alpha=0.6)
# frscc.set_title('Framed: correct by which guess')
# frscc.set_xlabel('By guess')
# frscc.set_xlim(1, 6)
# frscc.set_ylim(0, 1)
# frscc.yaxis.set_major_formatter(pct)

#%% Over time: Framed

fig, ax = plt.subplots(1, 1, figsize=(13, 8))
framed_results[['Sami', 'Laura']].resample('14D').mean().plot(ax=ax)
ax.set_title('Framed: scores over time (rolling average)')
ax.set_ylim(1, 6)
plt.tight_layout()
plt.savefig('Outputs\\Framed_overtime.png')

#%% Scoring: Waffle

waffle_labels = ['X', 0, 1, 2, 3, 4, 5]

waffle_scores = make_scores(waffle_results)
plot_bar(waffle_scores, name_str='Waffle', xticklabels=waffle_labels)
plt.tight_layout()
plt.savefig('Outputs\\Waffle_guess.png')

wfscc = waffle_scores.sort_index(ascending=False).cumsum().plot(color=colors, figsize=(8, 8), alpha=0.6)
wfscc.set_title('Waffle: correct by which guess')
wfscc.invert_xaxis()
wfscc.set_xlabel('By guess')
wfscc.set_xticklabels([0] + waffle_labels)
# wfscc.set_xlim(6, 0)
wfscc.set_ylim(0, 1)
wfscc.yaxis.set_major_formatter(pct)
plt.tight_layout()
plt.savefig('Outputs\\Waffle_cumu.png')

#%% Over time: Waffle

fig, ax = plt.subplots(1, 1, figsize=(13, 8))
waffle_results[['Sami', 'Laura']].resample('14D').mean().plot(ax=ax)
ax.set_title('Waffle: scores over time (rolling average)')
# ax.set_ylim(1, 6)
plt.tight_layout()
plt.savefig('Outputs\\Waffle_overtime.png')

#%% Scoring: Wordle

wordle_scores = make_scores(wordle_results)

plot_bar(wordle_scores, name_str='Wordle', xticklabels=[2, 3, 4, 5, 6, 'X'])
plt.tight_layout()
plt.savefig('Outputs\\Wordle_guess.png')

plot_cumu(wordle_scores, name_str='Wordle')
plt.tight_layout()
plt.savefig('Outputs\\Wordle_cumu.png')

#%% Scoring: Octordle

octo_score_list = []
for p in person:
    slc = octordle_results.loc[octordle_results['person']==p, 'max'].value_counts().sort_index()
    octo_score_list.append(slc)
octo_scores = pd.concat(octo_score_list, axis=1)
octo_scores.columns = person
octo_scores = octo_scores / octo_scores.sum()
octo_scores.index.name = None

plot_bar(octo_scores, name_str='Octordle', xticklabels=[10, 11, 12 ,13, 'X'])
plt.tight_layout()
plt.savefig('Outputs\\Octordle_guess.png')

plot_cumu(octo_scores, name_str='Octordle', xmin=10, xmax=14)
plt.xticks(plt.xticks()[0], [10, '', 11, '', 12, '', 13, '', 'X'])
plt.tight_layout()
plt.savefig('Outputs\\Octordle_cumu.png')

#%% Overall participation

to_total = ['Connections', 'Flagle', 'Framed', 'Octordle', 'Waffle', 'Wordle',
            'Worldle']
min_date = pd.Timestamp.today()
max_date = pd.Timestamp('2022-01-01')
for game in to_total:
    if game in ['Connections', 'Octordle', 'Worldle']:
        mi = game_results[game]['date'].min()
        if mi < min_date:
            min_date = mi
        ma = game_results[game]['date'].max()
        if ma > max_date:
            max_date = ma

totals = {p: pd.DataFrame(index=pd.date_range(min_date, max_date)) for p in person}

for game in to_total:
    if game in ['Flagle', 'Framed', 'Waffle', 'Wordle']:
        review_df = game_results[game].copy()
        for p in person:
            review_col = f'{p}_c'
            flag_col = f'{p}_NA'
            review_df[review_col] = review_df[p].between(1, 6)
            review_df[review_col] = review_df[review_col].astype(int)
            review_df[flag_col] = review_df[p].isna()
            review_df.loc[review_df[flag_col], review_col] = None
            current_cols = list(totals[p].columns)
            totals[p] = totals[p].join(review_df[review_col])
            totals[p].columns = current_cols + [game]
    if game in ['Connections', 'Octordle', 'Worldle']:
        for p in person:
            review_df = game_results[game][['date', 'person', 'complete']].copy()
            review_df = review_df[review_df['person']==p].set_index('date')['complete']
            review_df = review_df.astype(int)
            current_cols = list(totals[p].columns)
            totals[p] = totals[p].join(review_df)
            totals[p].columns = current_cols + [game]

for p in person:
    totals[p].to_csv(f'AmendedData\\{p}_participation.csv')

#%% Plot overall

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
for p, ax in zip(person, axs):
    total_p = totals[p].loc[start:end].copy()
    total_p.index = pd.to_datetime(total_p.index).date  # to make date fmt not weird
    sns.heatmap(total_p, cmap=['#ff6962', '#7abd91'], ax=ax,
                cbar=False)
    ax.set_facecolor('#ffffff')
    ax.set_title(p, fontsize=20)
    ax.set_xlabel(None)
plt.suptitle("Last 6 months of 'dles", fontsize=35)
plt.tight_layout()
plt.savefig('Outputs\\total_participation.png')

#%% Overall scores

scores = {p: pd.DataFrame(index=pd.date_range(min_date, max_date)) for p in person}

for game in to_total:
    if game in ['Flagle', 'Framed', 'Waffle', 'Wordle']:
        review_df = game_results[game].copy()
        for p in person:
            current_cols = list(scores[p].columns)
            scores[p] = scores[p].join(review_df[p])
            scores[p].columns = current_cols + [game]
    if game == 'Connections':
        review_df = game_results[game][['date', 'person'] + conn_c].copy()
        review_df[game] = review_df[conn_c].max(axis=1)
        for p in person:
            slc = review_df.loc[review_df['person']==p, ['date', game]]
            slc = slc.set_index('date')
            scores[p] = scores[p].join(slc)
    if game == 'Octordle':
        review_df = game_results[game][['date', 'person', 'max']].copy()
        review_df[game] = review_df['max']
        for p in person:
            slc = review_df.loc[review_df['person']==p, ['date', game]]
            slc = slc.set_index('date')
            scores[p] = scores[p].join(slc)
    if game == 'Worldle':
        review_df = game_results[game][['date', 'person', 'score']].copy()
        review_df[game] = review_df['score']
        for p in person:
            slc = review_df.loc[review_df['person']==p, ['date', game]]
            slc = slc.set_index('date')
            scores[p] = scores[p].join(slc)

for p in person:
    scores[p].to_csv(f'AmendedData\\{p}_scores.csv')

score_map = {'Connections': {'top': 4, 'bottom': 7},
             'Flagle': {'top': 1, 'bottom': 6},
             'Framed': {'top': 1, 'bottom': 6},
             'Octordle': {'top': 10, 'bottom': 13},
             'Waffle': {'top': 5, 'bottom': 0},
             'Wordle': {'top': 1, 'bottom': 6},
             'Worldle': {'top': 1, 'bottom': 6}}

ratings = {k: v.copy() for k, v in scores.items()}
for game in to_total:
    bottom = score_map[game]['bottom']
    top = score_map[game]['top']
    for p in person:
        # Min-max scale
        ratings[p][game] = (bottom - ratings[p][game]) / (bottom - top)
for p in person:
    ratings[p][ratings[p]<0] = -0.5

for p in person:
    scores[p].to_csv(f'AmendedData\\{p}_ratings.csv')

#%% Plot ratings

fig, axs = plt.subplots(1, 2, sharey=True, figsize=(15, 8))
for p, ax in zip(person, axs):
    rating_p = ratings[p].loc[start:end].copy()
    rating_p.index = pd.to_datetime(rating_p.index).date  # to make date fmt not weird
    sns.heatmap(rating_p, ax=ax, cmap='jet')
    ax.set_facecolor('#ffffff')
    ax.set_title(p, fontsize=20)
    ax.set_xlabel(None)
plt.suptitle("Last 6 months of 'dles", fontsize=35)
plt.tight_layout()
plt.savefig('Outputs\\total_ratings.png')
