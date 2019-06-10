#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('hw3.ok')


# # Homework 3: Trump, Twitter, and Text
# 
# ## Due Date: Monday 2/25, 11:59 pm PST
# 
# Welcome to the third homework assignment of Data 100/200! In this assignment, we will work with Twitter data in order to analyze Donald Trump's tweets.
# 
# **Collaboration Policy**
# 
# Data science is a collaborative activity. While you may talk with others about the homework, we ask that you **write your solutions individually**. If you do discuss the assignments with others please **include their names** below.

# **Collaborators**: *list collaborators here*

# In[2]:


# Run this cell to set up your notebook
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import zipfile

# Ensure that Pandas shows at least 280 characters in columns, so we can see full tweets
pd.set_option('max_colwidth', 280)

get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
import seaborn as sns
sns.set()
sns.set_context("talk")
import re


# ### Score Breakdown
# 
# Question | Points
# --- | ---
# Question 1 | 2
# Question 2 | 1
# Question 3 | 2
# Question 4a | 1
# Question 4b | 2
# Question 4c | 2
# Question 5a | 1
# Question 5b | 1
# Question 5c | 1
# Question 5d | 2
# Question 5e | 2
# Question 6a | 1
# Question 6b | 1
# Total | 19

# ---
# # Part 1: Importing the Data
# 
# We will again use the `fetch_and_cache` utility to download the dataset. 

# In[3]:


# Download the dataset
from ds100_utils import fetch_and_cache
data_url = 'http://www.ds100.org/sp19/assets/datasets/hw3-realdonaldtrump_tweets.json.zip'
file_name = 'hw3-realdonaldtrump_tweets.json.zip'

dest_path = fetch_and_cache(data_url=data_url, file=file_name)
print(f'Located at {dest_path}')


# Now that we've downloaded the tweets, let's unzip them and load them into our notebook. Run the cell below to unzip and read tweets from the json file into a list named `all_tweets`.

# In[4]:


# Unzip the dataset
my_zip = zipfile.ZipFile(dest_path, 'r')
with my_zip.open('hw3-realdonaldtrump_tweets.json', 'r') as f:
    all_tweets = json.load(f)


# Here is what a typical tweet from `all_tweets` looks like:

# In[5]:


from pprint import pprint # to get a more easily-readable view.
pprint(all_tweets[-1])


# ### Question 1
# 
# Construct a DataFrame called `trump` containing data from all the tweets stored in `all_tweets`. The index of the DataFrame should be the ID of each tweet (looks something like `907698529606541312`). It should have these columns:
# 
# - `time`: The time the tweet was created encoded as a datetime object. (Use `pd.to_datetime` to encode the timestamp.)
# - `source`: The source device of the tweet.
# - `text`: The text of the tweet.
# - `retweet_count`: The retweet count of the tweet. 
# 
# Finally, **the resulting DataFrame should be sorted by the index.**
# 
# **Warning:** *Some tweets will store the text in the `text` field and other will use the `full_text` field.*
# 
# <!--
# BEGIN QUESTION
# name: q1
# points: 2
# -->

# In[6]:


# pd.to_datetime(all_tweets['created_at'])
id = [i['id'] for i in all_tweets]
time = [pd.to_datetime(i['created_at']) for i in all_tweets]
source = [i['source'] for i in all_tweets]
text = [i['text'] 
        if 'text' in i 
        else i['full_text'] 
        for i in all_tweets]
retweet_count = [i['retweet_count'] for i in all_tweets]
trump = pd.DataFrame({'time' : time, 'source' : source, 'text' : text, 'retweet_count' : retweet_count},
                    index = id)
trump.head()


# In[7]:


ok.grade("q1");


# ---
# # Part 2: Tweet Source Analysis
# 
# In the following questions, we are going to find out the charateristics of Trump tweets and the devices used for the tweets.
# 
# First let's examine the source field:

# In[8]:


trump['source'].unique()


# ## Question 2
# 
# Notice how sources like "Twitter for Android" or "Instagram" are surrounded by HTML tags. In the cell below, clean up the `source` field by removing the HTML tags from each `source` entry.
# 
# **Hints:** 
# * Use `trump['source'].str.replace` along with a regular expression.
# * You may find it helpful to experiment with regular expressions at [regex101.com](https://regex101.com/).
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 1
# -->

# In[9]:


## Uncomment and complete
# trump['source'] = ...
trump['source'] = trump['source'].str.replace(r"\<.*?\>", "")
trump['source'].head()


# In[10]:


ok.grade("q2");


# In the following plot, we see that there are two device types that are more commonly used than others.

# In[11]:


plt.figure(figsize=(8, 6))
trump['source'].value_counts().plot(kind="bar")
plt.ylabel("Number of Tweets")
plt.title("Number of Tweets by Source");


# ## Question 3
# 
# Now that we have cleaned up the `source` field, let's now look at which device Trump has used over the entire time period of this dataset.
# 
# To examine the distribution of dates we will convert the date to a fractional year that can be plotted as a distribution.
# 
# (Code borrowed from https://stackoverflow.com/questions/6451655/python-how-to-convert-datetime-dates-to-decimal-years)

# In[12]:


import datetime
def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length

trump['year'] = trump['time'].apply(year_fraction)


# Now, use `sns.distplot` to overlay the distributions of Trump's 2 most frequently used web technologies over the years. Your final plot should look like:
# 
# <img src="images/source_years_q3.png" width="600px" />
# 
# <!--
# BEGIN QUESTION
# name: q3
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[40]:


# ...
trump['source'].value_counts().head(2).index
# trump.head()
# ranking = trump.groupby('source')['retweet_count'].agg(sum).sort_values(ascending=False)
# top_two = ranking.head(2)
# top_two

# sns.distplot()

for thissource in trump['source'].value_counts().head(2).index:
    sns.distplot(trump[trump['source'] == thissource]['year'], label = thissource[12: ], hist = True)
plt.legend()
plt.title('Distributions of Tweet Sources Over Years')
    


# ## Question 4
# 
# 
# Is there a difference between Trump's tweet behavior across these devices? We will attempt to answer this question in our subsequent analysis.
# 
# First, we'll take a look at whether Trump's tweets from an Android device come at different times than his tweets from an iPhone. Note that Twitter gives us his tweets in the [UTC timezone](https://www.wikiwand.com/en/List_of_UTC_time_offsets) (notice the `+0000` in the first few tweets).

# In[14]:


for tweet in all_tweets[:3]:
    print(tweet['created_at'])


# We'll convert the tweet times to US Eastern Time, the timezone of New York and Washington D.C., since those are the places we would expect the most tweet activity from Trump.

# In[15]:


trump['est_time'] = (
    trump['time'].dt.tz_localize("UTC") # Set initial timezone to UTC
                 .dt.tz_convert("EST") # Convert to Eastern Time
)
trump.head()


# ### Question 4a
# 
# Add a column called `hour` to the `trump` table which contains the hour of the day as floating point number computed by:
# 
# $$
# \text{hour} + \frac{\text{minute}}{60} + \frac{\text{second}}{60^2}
# $$
# 
# * **Hint:** See the cell above for an example of working with [dt accessors](https://pandas.pydata.org/pandas-docs/stable/getting_started/basics.html#basics-dt-accessors).
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 1
# -->

# In[35]:


# trump['est_time'].head().values
hour = trump['est_time'].dt.hour
minute = trump['est_time'].dt.minute
second = trump['est_time'].dt.second
# trump['est_time'].head()
trump['hour'] = hour + (minute / 60) + (second / (60 * 60))
# trump['hour']


# In[36]:


ok.grade("q4a");


# ### Question 4b
# 
# Use this data along with the seaborn `distplot` function to examine the distribution over hours of the day in eastern time that trump tweets on each device for the 2 most commonly used devices.  Your plot should look similar to the following:
# 
# <img src="images/device_hour4b.png" width="600px" />
# 
# <!--
# BEGIN QUESTION
# name: q4b
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[39]:


### make your plot here
# ...
for thissource in trump['source'].value_counts().head(2).index:
    sns.distplot(trump[trump['source'] == thissource]['hour'], label = thissource[12: ], hist = False)
plt.legend()
plt.title('Distributions of Tweet Hours for Different Tweet Sources')
plt.ylabel('fraction')


# ### Question 4c
# 
# According to [this Verge article](https://www.theverge.com/2017/3/29/15103504/donald-trump-iphone-using-switched-android), Donald Trump switched from an Android to an iPhone sometime in March 2017.
# 
# Let's see if this information significantly changes our plot. Create a figure similar to your figure from question 4b, but this time, only use tweets that were tweeted before 2017. Your plot should look similar to the following:
# 
# <img src="images/device_hour4c.png" width="600px" />
# 
# <!--
# BEGIN QUESTION
# name: q4c
# points: 2
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[57]:


### make your plot here
# ...

pre_2017 = trump[trump['est_time'].dt.year < 2017]
# trump[trump['source'] == thissource]['est_time'].dt.year < 2017

for thissource in pre_2017['source'].value_counts().head(2).index:
    sns.distplot(pre_2017[pre_2017['source'] == thissource]['hour'], label = thissource[12: ], hist = False)
plt.legend()
plt.title('Distributions of Tweet Hours for Different Tweet Sources (pre-2017)')
plt.ylabel('fraction')


# ### Question 4d
# 
# During the campaign, it was theorized that Donald Trump's tweets from Android devices were written by him personally, and the tweets from iPhones were from his staff. Does your figure give support to this theory? What kinds of additional analysis could help support or reject this claim?
# 
# <!--
# BEGIN QUESTION
# name: q4d
# points: 1
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# Yes, i think the figure supports this theory. If Trump tweet himself, it must be before he go to work and after his work. We can see from the plot, Android phone has a higher value around 9:00 and 22:00, which is about the time before and after work. Assitional analysis that could help can be the specific time he go to work and finish work. 

# ---
# # Part 3: Sentiment Analysis
# 
# It turns out that we can use the words in Trump's tweets to calculate a measure of the sentiment of the tweet. For example, the sentence "I love America!" has positive sentiment, whereas the sentence "I hate taxes!" has a negative sentiment. In addition, some words have stronger positive / negative sentiment than others: "I love America." is more positive than "I like America."
# 
# We will use the [VADER (Valence Aware Dictionary and sEntiment Reasoner)](https://github.com/cjhutto/vaderSentiment) lexicon to analyze the sentiment of Trump's tweets. VADER is a lexicon and rule-based sentiment analysis tool that is specifically attuned to sentiments expressed in social media which is great for our usage.
# 
# The VADER lexicon gives the sentiment of individual words. Run the following cell to show the first few rows of the lexicon:

# In[58]:


print(''.join(open("vader_lexicon.txt").readlines()[:10]))


# ## Question 5
# 
# As you can see, the lexicon contains emojis too! Each row contains a word and the *polarity* of that word, measuring how positive or negative the word is.
# 
# (How did they decide the polarities of these words? What are the other two columns in the lexicon? See the link above.)
# 
# ### Question 5a
# 
# Read in the lexicon into a DataFrame called `sent`. The index of the DataFrame should be the words in the lexicon. `sent` should have one column named `polarity`, storing the polarity of each word.
# 
# * **Hint:** The `pd.read_csv` function may help here. 
# 
# <!--
# BEGIN QUESTION
# name: q5a
# points: 1
# -->

# In[61]:


# sent = ...
sent = pd.read_csv('vader_lexicon.txt', sep = '\t', usecols = [0, 1], 
                   names = ['token', 'polarity'], index_col = 'token')
sent.head()


# In[62]:


ok.grade("q5a");


# ### Question 5b
# 
# Now, let's use this lexicon to calculate the overall sentiment for each of Trump's tweets. Here's the basic idea:
# 
# 1. For each tweet, find the sentiment of each word.
# 2. Calculate the sentiment of each tweet by taking the sum of the sentiments of its words.
# 
# First, let's lowercase the text in the tweets since the lexicon is also lowercase. Set the `text` column of the `trump` DataFrame to be the lowercased text of each tweet.
# 
# <!--
# BEGIN QUESTION
# name: q5b
# points: 1
# -->

# In[64]:


# ...
trump['text'] = trump['text'].str.lower()
trump.head()


# In[65]:


ok.grade("q5b");


# ### Question 5c
# 
# Now, let's get rid of punctuation since it will cause us to fail to match words. Create a new column called `no_punc` in the `trump` DataFrame to be the lowercased text of each tweet with all punctuation replaced by a single space. We consider punctuation characters to be *any character that isn't a Unicode word character or a whitespace character*. You may want to consult the Python documentation on regexes for this problem.
# 
# (Why don't we simply remove punctuation instead of replacing with a space? See if you can figure this out by looking at the tweet data.)
# 
# <!--
# BEGIN QUESTION
# name: q5c
# points: 1
# -->

# In[71]:


# Save your regex in punct_re
punct_re = r'[^\w\s+]'
# trump['no_punc'] = ...
trump['no_punc'] = trump['text'].str.replace(punct_re, ' ')
trump.head()


# In[72]:


ok.grade("q5c");


# ### Question 5d
# 
# Now, let's convert the tweets into what's called a [*tidy format*](https://cran.r-project.org/web/packages/tidyr/vignettes/tidy-data.html) to make the sentiments easier to calculate. Use the `no_punc` column of `trump` to create a table called `tidy_format`. The index of the table should be the IDs of the tweets, repeated once for every word in the tweet. It has two columns:
# 
# 1. `num`: The location of the word in the tweet. For example, if the tweet was "i love america", then the location of the word "i" is 0, "love" is 1, and "america" is 2.
# 2. `word`: The individual words of each tweet.
# 
# The first few rows of our `tidy_format` table look like:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>num</th>
#       <th>word</th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>894661651760377856</th>
#       <td>0</td>
#       <td>i</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>1</td>
#       <td>think</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>2</td>
#       <td>senator</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>3</td>
#       <td>blumenthal</td>
#     </tr>
#     <tr>
#       <th>894661651760377856</th>
#       <td>4</td>
#       <td>should</td>
#     </tr>
#   </tbody>
# </table>
# 
# **Note that your DataFrame may look different from the one above.** However, you can double check that your tweet with ID `894661651760377856` has the same rows as ours. Our tests don't check whether your table looks exactly like ours.
# 
# As usual, try to avoid using any for loops. Our solution uses a chain of 5 methods on the `trump` DataFrame, albeit using some rather advanced Pandas hacking.
# 
# * **Hint 1:** Try looking at the `expand` argument to pandas' `str.split`.
# 
# * **Hint 2:** Try looking at the `stack()` method.
# 
# * **Hint 3:** Try looking at the `level` parameter of the `reset_index` method.
# 
# <!--
# BEGIN QUESTION
# name: q5d
# points: 2
# -->

# In[89]:


# tidy_format = ...
tidy_format = trump['no_punc'].str.split(expand = True).stack().reset_index(level = 1).rename(columns = {'level_1': 'num', 0: 'word'})
tidy_format.head()


# In[90]:


ok.grade("q5d");


# ### Question 5e
# 
# Now that we have this table in the tidy format, it becomes much easier to find the sentiment of each tweet: we can join the table with the lexicon table. 
# 
# Add a `polarity` column to the `trump` table.  The `polarity` column should contain the sum of the sentiment polarity of each word in the text of the tweet.
# 
# **Hints:** 
# * You will need to merge the `tidy_format` and `sent` tables and group the final answer.
# * If certain words are not found in the `sent` table, set their polarities to 0.
# 
# <!--
# BEGIN QUESTION
# name: q5e
# points: 2
# -->

# In[105]:


trump['polarity'] = tidy_format.merge(sent, how = 'outer', left_on = 'word', right_index = True).reset_index().groupby('index')['polarity'].sum()


# In[106]:


ok.grade("q5e");


# Now we have a measure of the sentiment of each of his tweets! Note that this calculation is rather basic; you can read over the VADER readme to understand a more robust sentiment analysis.
# 
# Now, run the cells below to see the most positive and most negative tweets from Trump in your dataset:

# In[107]:


print('Most negative tweets:')
for t in trump.sort_values('polarity').head()['text']:
    print('\n  ', t)


# In[108]:


print('Most positive tweets:')
for t in trump.sort_values('polarity', ascending=False).head()['text']:
    print('\n  ', t)


# ## Question 6
# 
# Now, let's try looking at the distributions of sentiments for tweets containing certain keywords.
# 
# ### Question 6a
# 
# In the cell below, create a single plot showing both the distribution of tweet sentiments for tweets containing `nytimes`, as well as the distribution of tweet sentiments for tweets containing `fox`.
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# manual: true
# -->
# <!-- EXPORT TO PDF -->

# In[127]:


# ...
sns.distplot(trump[trump['text'].str.contains('china')]['polarity'], label = 'china')
sns.distplot(trump[trump['text'].str.contains('japan')]['polarity'], label = 'japan')
plt.legend()
plt.title('Distribution of Tweet Sentiments')


# ### Question 6b
# Comment on what you observe in the plot above. Can you find other pairs of keywords that lead to interesting plots? (If you modify your code in 6a, remember to change the words back to `nytimes` and `fox` before submitting for grading).
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# manual: true
# --><!-- EXPORT TO PDF -->

# We can see that in general, the tweets containig 'fox' have higher sentiment than those containing 'nytimes'. 
# For other pairs of keywords, for example: (china and japan) we can see that both of them have high polarity in average.

# In[128]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('hw3.ipynb', 'hw3.pdf')
ok.submit()


# In[ ]:




