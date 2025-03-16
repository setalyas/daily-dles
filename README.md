# Daily 'dles
Data visualisations from the "daily 'dles" (daily trivia games)

Overall process:

- Export message data from Whatsapp (extract directly from phone), that contains daily scores
- Scrape scores out of the messages
- Visualise

### Worldle deep dive

To be able to further analyse Worldle scores (e.g. average scores by continent), need a list of what each Worldle answer was per day.

I tried emailing Teuteuf, but despite responding to the query overall, they never sent a list of results for the different games per day.

Thoughts to access:

- I thought I could get from Worldle directly. But that would involve paying for premium -- and even then, I'd have to replay each day individually to get the results. Doesn't scale well for multiple years of scores.
- Archive.org has hits for some Worldle days. Two issues: it doesn't have all days; again, would have to replay each day to get the results.
- I tried looking for [lists of Worldle answers through Google](https://www.google.com/search?q="worldle"+previous+answers). There are some lists:
  - https://www.dexerto.com/gaming/daily-worldle-answers-clues-and-tips-for-todays-worldle-country-of-the-day-1979343/ and others
  - https://www.gamespew.com/2025/02/worldle-answer/ and others
  - https://world3dmap.com/worldle-answers/
  - https://x.com/Worldle_Game
- I tried hunting on Reddit, but despite some similar posts asking for archives ([e.g.](https://www.reddit.com/r/Worldle/comments/sxg5vv/worldle_archive/)), nothing useful was uncovered.
- I hunted for Github repos for similar visualisation attempts. Nothing turned up, but...
  - I found https://github.com/sundehakon/WorldleSolve, which links to https://sundehakon.tech/tools, which has a tool that goes back at least to the start of the year.
  - The tool breaks when going back into 2024 and further. _However_, looking into the Network tab on Dev Tools, can see JSONs being scraped back further with valid data, despite the tool not displaying them correctly.
  - By attempting different queries, can get back to https://teuteuf-dashboard-assets.pages.dev/data/worldle/games/2022/2022-01-22.json.

I cross-checked this against [other sources I had done spot checks on](https://docs.google.com/spreadsheets/d/1j--CtVpIvrVYOah1hNaM4obJi6UcCN4Gm8Zkxe25D9Y/edit?gid=0#gid=0) (GameSpew, Dexerto, Wayback Machine, Worldle official twitter) and it passed a lot of spot checks.

### Country info

I tried grabbing info from Wikipedia tables, but it was mostly by country name (which needs cleaning and attention), not ISO-3166 codes.

=> found https://simplemaps.com/data/countries, which has a simple clean dataset for the main metrics (area, population, continent)