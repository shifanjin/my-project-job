#!/usr/bin/env python
# coding: utf-8

# In[396]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj1.ok')


# # Project 1: Food Safety 
# ## Cleaning and Exploring Data with Pandas
# ## Due Date: Tuesday 2/12, 6:00 PM
# ## Collaboration Policy
# 
# Data science is a collaborative activity. While you may talk with others about
# the project, we ask that you **write your solutions individually**. If you do
# discuss the assignments with others please **include their names** at the top
# of your notebook.

# **Collaborators**: *list collaborators here*

# 
# ## This Assignment
# <img src="scoreCard.jpg" width=400>
# 
# In this project, you will investigate restaurant food safety scores for restaurants in San Francisco. Above is a sample score card for a restaurant. The scores and violation information have been made available by the San Francisco Department of Public Health. The main goal for this assignment is to understand how restaurants are scored. We will walk through various steps of exploratory data analysis to do this. We will provide comments and insights along the way to give you a sense of how we arrive at each discovery and what next steps it leads to.
# 
# As we clean and explore these data, you will gain practice with:
# * Reading simple csv files
# * Working with data at different levels of granularity
# * Identifying the type of data collected, missing values, anomalies, etc.
# * Applying probability sampling techniques
# * Exploring characteristics and distributions of individual variables
# 
# ## Score Breakdown
# Question | Points
# --- | ---
# 1a | 1
# 1b | 0
# 1c | 0
# 1d | 3
# 1e | 1
# 2a | 1
# 2b | 2
# 3a | 2
# 3b | 0
# 3c | 2
# 3d | 1
# 3e | 1
# 3f | 1
# 4a | 1
# 4b | 1
# 4c | 1
# 4d | 1
# 4e | 1
# 4f | 1
# 4g | 2
# 4h | 1
# 4i | 1
# 5a | 2
# 5b | 3
# 6a | 1
# 6b | 1
# 6c | 1
# 7a | 2
# 7b | 3
# 7c | 3
# 8a | 2
# 8b | 2
# 8c | 6
# 8d | 2
# 8e | 3
# Total | 56

# To start the assignment, run the cell below to set up some imports and the automatic tests that we will need for this assignment:
# 
# In many of these assignments (and your future adventures as a data scientist) you will use `os`, `zipfile`, `pandas`, `numpy`, `matplotlib.pyplot`, and optionally `seaborn`.  
# 
# 1. Import each of these libraries `as` their commonly used abbreviations (e.g., `pd`, `np`, `plt`, and `sns`).  
# 1. Don't forget to include `%matplotlib inline` which enables [inline matploblib plots](http://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-matplotlib). 
# 1. If you want to use `seaborn`, add the line `sns.set()` to make your plots look nicer.

# In[397]:


#...
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import zipfile
import seaborn as sns
from pathlib import Path

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# In[398]:


import sys

assert 'zipfile'in sys.modules
assert 'pandas'in sys.modules and pd
assert 'numpy'in sys.modules and np
assert 'matplotlib'in sys.modules and plt


# ## Downloading the Data
# 
# For this assignment, we need this data file: http://www.ds100.org/sp19/assets/datasets/proj1-SFBusinesses.zip
# 
# We could write a few lines of code that are built to download this specific data file, but it's a better idea to have a general function that we can reuse for all of our assignments. Since this class isn't really about the nuances of the Python file system libraries, we've provided a function for you in ds100_utils.py called `fetch_and_cache` that can download files from the internet.
# 
# This function has the following arguments:
# - data_url: the web address to download
# - file: the file in which to save the results
# - data_dir: (default="data") the location to save the data
# - force: if true the file is always re-downloaded 
# 
# The way this function works is that it checks to see if `data_dir/file` already exists. If it does not exist already or if `force=True`, the file at `data_url` is downloaded and placed at `data_dir/file`. The process of storing a data file for reuse later is called caching. If `data_dir/file` already and exists `force=False`, nothing is downloaded, and instead a message is printed letting you know the date of the cached file.
# 
# The function returns a `pathlib.Path` object representing the location of the file ([pathlib docs](https://docs.python.org/3/library/pathlib.html#basic-use)). 

# In[399]:


import ds100_utils
source_data_url = 'http://www.ds100.org/sp19/assets/datasets/proj1-SFBusinesses.zip'
target_file_name = 'data.zip'

# Change the force=False -> force=True in case you need to force redownload the data
dest_path = ds100_utils.fetch_and_cache(
    data_url=source_data_url, 
    data_dir='.', 
    file=target_file_name, 
    force=False)


# After running the cell above, if you list the contents of the directory containing this notebook, you should see `data.zip`.

# In[400]:


get_ipython().system('ls')


# ---
# ## 1: Loading Food Safety Data
# 
# We have data, but we don't have any specific questions about the data yet, so let's focus on understanding the structure of the data. This involves answering questions such as:
# 
# * Is the data in a standard format or encoding?
# * Is the data organized in records?
# * What are the fields in each record?
# 
# Let's start by looking at the contents of `data.zip`. It's not just a single file, but a compressed directory of multiple files. We could inspect it by uncompressing it using a shell command such as `!unzip data.zip`, but in this project we're going to do almost everything in Python for maximum portability.

# ### Question 1a: Looking Inside and Extracting the Zip Files
# 
# Assign `my_zip` to a `Zipfile.zipfile` object representing `data.zip`, and 1ssign `list_files` to a list of all the names of the files in `data.zip`.
# 
# *Hint*: The [Python docs](https://docs.python.org/3/library/zipfile.html) describe how to create a `zipfile.ZipFile` object. You might also look back at the code from lecture and lab. It's OK to copy and paste code from previous assignments and demos, though you might get more out of this exercise if you type out an answer.
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[401]:


my_zip = zipfile.ZipFile(file = dest_path, mode = 'r') 
#my_zip.extractall('data')
data_dir_path = Path('data') # creates a Path object that points to the data directory
list_names = [x.name for x in data_dir_path.glob('*') if x.is_file()]
list_names


# In[402]:


ok.grade("q1a");


# In your answer above, if you have written something like `zipfile.ZipFile('data.zip', ...)`, we suggest changing it to read `zipfile.ZipFile(dest_path, ...)`. In general, we **strongly suggest having your filenames hard coded as string literals only once** in a notebook. It is very dangerous to hard code things twice, because if you change one but forget to change the other, you can end up with bugs that are very hard to find.

# Now display the files' names and their sizes.
# 
# If you're not sure how to proceed, read about the attributes of a `ZipFile` object in the Python docs linked above.

# In[11]:


...


# Often when working with zipped data, we'll never unzip the actual zipfile. This saves space on our local computer. However, for this project, the files are small, so we're just going to unzip everything. This has the added benefit that you can look inside the csv files using a text editor, which might be handy for understanding what's going on. The cell below will unzip the csv files into a subdirectory called `data`. Just run it.

# In[403]:


from pathlib import Path
data_dir = Path('data')
my_zip.extractall(data_dir)
get_ipython().system('ls {data_dir}')


# The cell above created a folder called `data`, and in it there should be four CSV files. Open up `legend.csv` to see its contents. You should see something that looks like:
# 
#     "Minimum_Score","Maximum_Score","Description"
#     0,70,"Poor"
#     71,85,"Needs Improvement"
#     86,90,"Adequate"
#     91,100,"Good"

# ### Question 1b: Programatically Looking Inside the Files

# The `legend.csv` file does indeed look like a well-formed CSV file. Let's check the other three files. Rather than opening up each file manually, let's use Python to print out the first 5 lines of each. The `ds100_utils` library has a method called `head` that will allow you to retrieve the first N lines of a file as a list. For example `ds100_utils.head('data/legend.csv', 5)` will return the first 5 lines of "data/legend.csv". Try using this function to print out the first 5 lines of all four files that we just extracted from the zipfile.

# In[404]:


#...
ds100_utils.head('data/businesses.csv', 5)
ds100_utils.head('data/inspections.csv', 5)
ds100_utils.head('data/legend.csv', 5)
ds100_utils.head('data/violations.csv', 5)


# ### Question 1c: Reading in the Files
# 
# Based on the above information, let's attempt to load `businesses.csv`, `inspections.csv`, and `violations.csv` into pandas data frames with the following names: `bus`, `ins`, and `vio` respectively.
# 
# *Note:* Because of character encoding issues one of the files (`bus`) will require an additional argument `encoding='ISO-8859-1'` when calling `pd.read_csv`. One day you should read all about [character encodings](https://www.diveinto.org/python3/strings.html).

# In[405]:


# path to directory containing data
dsDir = Path('data')

bus = pd.read_csv("data/businesses.csv", encoding='ISO-8859-1')
ins = pd.read_csv("data/inspections.csv")
vio = pd.read_csv("data/violations.csv")


# Now that you've read in the files, let's try some `pd.DataFrame` methods ([docs](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.html)).
# Use the `DataFrame.head` method to show the top few lines of the `bus`, `ins`, and `vio` dataframes. Use `Dataframe.describe` to learn about the numeric columns.

# In[406]:


#...
bus.head()
ins.head()
vio.head()


# The `DataFrame.describe` method can also be handy for computing summaries of various statistics of our dataframes. Try it out with each of our 3 dataframes.

# In[409]:


#...
bus.describe()
ins.describe()
vio.describe()
len(bus)


# Now, we perform some sanity checks for you to verify that you loaded the data with the right structure. Run the following cells to load some basic utilities (you do not need to change these at all):

# First, we check the basic structure of the data frames you created:

# In[410]:


assert all(bus.columns == ['business_id', 'name', 'address', 'city', 'state', 'postal_code',
                           'latitude', 'longitude', 'phone_number'])
assert 6400 <= len(bus) <= 6420

assert all(ins.columns == ['business_id', 'score', 'date', 'type'])
assert 14210 <= len(ins) <= 14250

assert all(vio.columns == ['business_id', 'date', 'description'])
assert 39020 <= len(vio) <= 39080


# Next we'll check that the statistics match what we expect. The following are hard-coded statistical summaries of the correct data.

# In[411]:


bus_summary = pd.DataFrame(**{'columns': ['business_id', 'latitude', 'longitude'],
 'data': {'business_id': {'50%': 68294.5, 'max': 94574.0, 'min': 19.0},
  'latitude': {'50%': 37.780435, 'max': 37.824494, 'min': 37.668824},
  'longitude': {'50%': -122.41885450000001,
   'max': -122.368257,
   'min': -122.510896}},
 'index': ['min', '50%', 'max']})

ins_summary = pd.DataFrame(**{'columns': ['business_id', 'score'],
 'data': {'business_id': {'50%': 61462.0, 'max': 94231.0, 'min': 19.0},
  'score': {'50%': 92.0, 'max': 100.0, 'min': 48.0}},
 'index': ['min', '50%', 'max']})

vio_summary = pd.DataFrame(**{'columns': ['business_id'],
 'data': {'business_id': {'50%': 62060.0, 'max': 94231.0, 'min': 19.0}},
 'index': ['min', '50%', 'max']})

from IPython.display import display

print('What we expect from your Businesses dataframe:')
display(bus_summary)
print('What we expect from your Inspections dataframe:')
display(ins_summary)
print('What we expect from your Violations dataframe:')
display(vio_summary)


# The code below defines a testing function that we'll use to verify that your data has the same statistics as what we expect. Run these cells to define the function. The `df_allclose` function has this name because we are verifying that all of the statistics for your dataframe are close to the expected values. Why not `df_allequal`? It's a bad idea in almost all cases to compare two floating point values like 37.780435, as rounding error can cause spurious failures.

# ## Question 1d: Verifying the data
# 
# Now let's run the automated tests. If your dataframes are correct, then the following cell will seem to do nothing, which is a good thing! However, if your variables don't match the correct answers in the main summary statistics shown above, an exception will be raised.
# 
# <!--
# BEGIN QUESTION
# name: q1d
# points: 3
# -->

# In[412]:


"""Run this cell to load this utility comparison function that we will use in various
tests below (both tests you can see and those we run internally for grading).

Do not modify the function in any way.
"""


def df_allclose(actual, desired, columns=None, rtol=5e-2):
    """Compare selected columns of two dataframes on a few summary statistics.
    
    Compute the min, median and max of the two dataframes on the given columns, and compare
    that they match numerically to the given relative tolerance.
    
    If they don't match, an AssertionError is raised (by `numpy.testing`).
    """    
    # summary statistics to compare on
    stats = ['min', '50%', 'max']
    
    # For the desired values, we can provide a full DF with the same structure as
    # the actual data, or pre-computed summary statistics.
    # We assume a pre-computed summary was provided if columns is None. In that case, 
    # `desired` *must* have the same structure as the actual's summary
    if columns is None:
        des = desired
        columns = desired.columns
    else:
        des = desired[columns].describe().loc[stats]

    # Extract summary stats from actual DF
    act = actual[columns].describe().loc[stats]

    return np.allclose(act, des, rtol)


# In[413]:


ok.grade("q1d");


# ### Question 1e: Identifying Issues with the Data

# Use the `head` command on your three files again. This time, describe at least one potential problem with the data you see. Consider issues with missing values and bad data.
# 
# <!--
# BEGIN QUESTION
# name: q1e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# In bus.head(), we can see that the phone_number column has NaN values.

# We will explore each file in turn, including determining its granularity and primary keys and exploring many of the variables individually. Let's begin with the businesses file, which has been read into the `bus` dataframe.

# ---
# ## 2: Examining the Business Data
# 
# From its name alone, we expect the `businesses.csv` file to contain information about the restaurants. Let's investigate the granularity of this dataset.
# 
# **Important note: From now on, the local autograder tests will not be comprehensive. You can pass the automated tests in your notebook but still fail tests in the autograder.** Please be sure to check your results carefully.

# ### Question 2a
# 
# Examining the entries in `bus`, is the `business_id` unique for each record? Your code should compute the answer, i.e. don't just hard code `True` or `False`.
# 
# Hint: use `value_counts()` or `unique()` to determine if the `business_id` series has any duplicates.
# 
# <!--
# BEGIN QUESTION
# name: q2a
# points: 1
# -->

# In[414]:


helper = bus['business_id'].value_counts().unique()
if len(helper) == 1 :
    is_business_id_unique = True
else:
    is_business_id_unique  = False
is_business_id_unique


# In[415]:


ok.grade("q2a");


# ### Question 2b
# 
# With this information, you can address the question of granularity. Answer the questions below.
# 
# 1. What does each record represent (e.g., a business, a restaurant, a location, etc.)?  
# 1. What is the primary key?
# 1. What would you find by grouping by the following columns: `business_id`, `name`, `address`?
# 
# Please write your answer in the markdown cell below. You may create new cells below your answer to run code, but **please never add cells between a question cell and the answer cell below it.**
# 
# <!--
# BEGIN QUESTION
# name: q2b
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1. Each record represent a business. 
# 2. The primary key is the business_id.
# 3. Since business_id is unique, when grouping by 'business_id', there are 6406 records (same as the total number of rows). However, when grouping by 'name', there are 5758 records, and with 'address' there will be 5626 records. This might tell us that there are repeated names or address or missing values. 

# In[417]:


# use this cell for scratch work
#bus.head()
ins.head()
vio.head()

x = bus.groupby('address')
y = bus.groupby('name')
x.head()
len(x)
len(bus)

bus.head()
len(bus)


# ---
# ## 3: Zip Codes
# 
# Next, let's  explore some of the variables in the business table. We begin by examining the postal code.
# 
# ### Question 3a
# 
# Answer the following questions about the `postal code` column in the `bus` data frame?  
# 1. Are ZIP codes quantitative or qualitative? If qualitative, is it ordinal or nominal? 
# 1. What data type is used to represent a ZIP code?
# 
# *Note*: ZIP codes and postal codes are the same thing.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# 1. ZIP codes are quanlitative and nominal.
# 2. string

# ### Question 3b
# 
# How many restaurants are in each ZIP code? 
# 
# In the cell below, create a series where the index is the postal code and the value is the number of records with that postal code in descending order of count. 94110 should be at the top with a count of 596. 
# 
# <!--
# BEGIN QUESTION
# name: q3b
# points: 0
# -->

# In[418]:


zip_counts = bus['postal_code'].value_counts().sort_values(ascending=False)
zip_counts.head()


# Did you take into account that some businesses have missing ZIP codes?

# In[419]:


print('zip_counts describes', sum(zip_counts), 'records.')
print('The original data have', len(bus), 'records')


# Missing data is extremely common in real-world data science projects. There are several ways to include missing postal codes in the `zip_counts` series above. One approach is to use the `fillna` method of the series, which will replace all null (a.k.a. NaN) values with a string of our choosing. In the example below, we picked "?????". When you run the code below, you should see that there are 240 businesses with missing zip code.

# In[420]:


zip_counts = bus.fillna("?????").groupby("postal_code").size().sort_values(ascending=False)
zip_counts.head(15)


# An alternate approach is to use the DataFrame `value_counts` method with the optional argument `dropna=False`, which will ensure that null values are counted. In this case, the index will be `NaN` for the row corresponding to a null postal code.

# In[422]:


bus["postal_code"].value_counts(dropna=False).sort_values(ascending = False).head(15)
len(bus)


# Missing zip codes aren't our only problem. There are also some records where the postal code is wrong, e.g., there are 3 'Ca' and 3 'CA' values. Additionally, there are some extended postal codes that are 9 digits long, rather than the typical 5 digits.
# 
# Let's clean up the extended zip codes by dropping the digits beyond the first 5. Rather than deleting or replacing the old values in the `postal_code` columnm, we'll instead create a new column called `postal_code_5`.
# 
# The reason we're making a new column is that it's typically good practice to keep the original values when we are manipulating data. This makes it easier to recover from mistakes, and also makes it more clear that we are not working with the original raw data.

# In[424]:


bus['postal_code_5'] = bus['postal_code'].str[:5]
bus.head()
len(bus)


# ### Question 3c : A Closer Look at Missing ZIP Codes
# 
# Let's look more closely at records with missing ZIP codes. Describe why some records have missing postal codes.  Pay attention to their addresses. You will need to look at many entries, not just the first five.
# 
# *Hint*: The `isnull` method of a series returns a boolean series which is true only for entries in the original series that were missing.
# 
# <!--
# BEGIN QUESTION
# name: q3c
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# We can see that most of them have invalid locations and some of them don't have a phone_number. This might be due to not updating their information. 

# In[426]:


# You can use this cell as scratch to explore the data
null_rows = bus['postal_code'].isnull()
bus[null_rows]
bus.head(10)
len(bus)


# ### Question 3d: Incorrect ZIP Codes

# This dataset is supposed to be only about San Francisco, so let's set up a list of all San Francisco ZIP codes.

# In[482]:


all_sf_zip_codes = ["94102", "94103", "94104", "94105", "94107", "94108", 
                    "94109", "94110", "94111", "94112", "94114", "94115", 
                    "94116", "94117", "94118", "94119", "94120", "94121", 
                    "94122", "94123", "94124", "94125", "94126", "94127", 
                    "94128", "94129", "94130", "94131", "94132", "94133", 
                    "94134", "94137", "94139", "94140", "94141", "94142", 
                    "94143", "94144", "94145", "94146", "94147", "94151", 
                    "94158", "94159", "94160", "94161", "94163", "94164", 
                    "94172", "94177", "94188"]


# Set `weird_zip_code_businesses` equal to a new dataframe showing only rows corresponding to ZIP codes that are not valid and not missing. Use the `postal_code_5` column.
# 
# *Hint*: The `~` operator inverts a boolean array. Use in conjunction with `isin`.
# 
# <!--
# BEGIN QUESTION
# name: q3d1
# points: 0
# -->

# In[484]:


weird_row = ~bus['postal_code_5'].isin(all_sf_zip_codes)
weird_1 = bus[weird_row]
len(weird_1)
not_missing_rows = weird_1['postal_code_5'].notnull()
weird_zip_code_businesses = weird_1[not_missing_rows]
weird_zip_code_businesses
#len(bus)


# If we were doing very serious data analysis, we might indivdually look up every one of these strange records. Let's focus on just two of them: ZIP codes 94545 and 94602. Use a search engine to identify what cities these ZIP codes appear in. Try to explain why you think these two ZIP codes appear in your dataframe. For the one with ZIP code 94602, try searching for the business name and locate its real address.
# <!--
# BEGIN QUESTION
# name: q3d2
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# Zip code: 94545 corresponds to Hayward, CA
# Zip code: 94602 corresponds to Oakland, CA
# These two appears in my data frame because their zip codes aren't in all San Francisco ZIP codes (all_sf_zip_codes).

# ### Question 3e
# 
# We often want to clean the data to improve our analysis. This cleaning might include changing values for a variable or dropping records.
# 
# The value 94602 is wrong. Change it to the most reasonable correct value, using all information you have available. Modify the `postal_code_5` field using `bus['postal_code_5'].str.replace` to replace 94602.
# 
# <!--
# BEGIN QUESTION
# name: q3e
# points: 1
# -->

# In[485]:


# WARNING: Be careful when uncommenting the line below, it will set the entire column to NaN unless you 
# put something to the right of the ellipses.
bus['postal_code_5'] = bus['postal_code_5'].str.replace("94602", "94102")


# In[486]:


ok.grade("q3e");


# ### Question 3f
# 
# Now that we have corrected one of the weird postal codes, let's filter our `bus` data such that only postal codes from San Francisco remain. While we're at it, we'll also remove the businesses that are missing a postal code. As we mentioned in question 3d, filtering our postal codes in this way may not be ideal. (Fortunately, this is just a course assignment.)
# 
# Assign `bus` to a new dataframe that has the same columns but only the rows with ZIP codes in San Francisco.
# 
# <!--
# BEGIN QUESTION
# name: q3f
# points: 1
# -->

# In[488]:


not_null_rows = bus['postal_code_5'].notnull()
x = bus[not_null_rows]

sf_row = x['postal_code_5'].isin(all_sf_zip_codes)
bus = bus[sf_row]
len(bus)


#bus = bus[not_null_rows]
#bus.head()
#len(bus)


# In[441]:


ok.grade("q3f");


# ## 4: Sampling from the Business Data
# We can now sample from the business data using the cleaned ZIP code data. Make sure to use `postal_code_5` instead of `postal_code` for all parts of this question.

# ### Question 4a
# 
# First, complete the following function `sample`, which takes as arguments a series, `series`, and a sample size, `n`, and returns a simple random sample (SRS) of size `n` from the series. Recall that in SRS, sampling is performed **without** replacement. The result should be a **list** of the `n` values that are in the sample.
# 
# *Hint*: Consider using [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.14.1/reference/generated/numpy.random.choice.html).
# 
# <!--
# BEGIN QUESTION
# name: q4a
# points: 1
# -->

# In[442]:


def sample(series, n):
    # Do not change the following line of code in any way!
    # In case you delete it, it should be "np.random.seed(40)"
    np.random.seed(40)
    result = np.random.choice(series, n, replace = False)
    return result.tolist()
    
    #...
    


# In[443]:


ok.grade("q4a");


# ### Question 4b
# Suppose we take a SRS of 5 businesses from the business data. What is the probability that the business named AMERICANA GRILL & FOUNTAIN is in the sample?
# <!--
# BEGIN QUESTION
# name: q4b
# points: 1
# -->

# In[444]:


q4b_answer = 5/len(bus)
q4b_answer
len(bus)


# In[445]:


ok.grade("q4b");


# ### Question 4c
# Collect a stratified random sample of business names, where each stratum consists of a postal code. Collect one business name per stratum. Assign `bus_strat_sample` to a series of business names selected by this sampling procedure.
# 
# Hint: You can use the `sample` function you defined earlier.
# 
# <!--
# BEGIN QUESTION
# name: q4c
# points: 1
# -->

# In[446]:


x = bus[['name','postal_code_5']].groupby('postal_code_5')
y = x.agg(sample, 1)['name']
# to get the values! use the function lambda z: z[0] to access
bus_strat_sample = y.agg(lambda z: z[0])
#bus_strat_sample = y
bus_strat_sample.head()


# In[447]:


ok.grade("q4c");


# ### Question 4d
# 
# What is the probability that AMERICANA GRILL & FOUNTAIN is selected as part of this stratified random sampling procedure?
# <!--
# BEGIN QUESTION
# name: q4d
# points: 1
# -->

# In[448]:


q4d_answer = 1/160
q4d_answer


# In[449]:


ok.grade("q4d");


# ### Question 4e
# Collect a cluster sample of business IDs, where each cluster is a postal code, with 5 clusters in the sample. Assign `bus_cluster_sample` to a series of business IDs selected by this sampling procedure.
# 
# Hint: Consider using [`isin`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.isin.html).
# 
# <!--
# BEGIN QUESTION
# name: q4e
# points: 1
# -->

# In[450]:


cluster_sample = sample(bus['postal_code_5'], n = 5)
the_rows = bus['postal_code_5'].isin(cluster_sample)
sample_ids = bus[the_rows]['business_id']
bus_cluster_sample = sample_ids
bus_cluster_sample.head()
bus[the_rows]


# In[451]:


ok.grade("q4e");


# ### Question 4f
# What is the probability that AMERICANA GRILL & FOUNTAIN is selected as part of this cluster sampling procedure?
# <!--
# BEGIN QUESTION
# name: q4f
# points: 1
# -->

# In[112]:


q4f_answer = 5/30
q4f_answer


# In[113]:


ok.grade("q4f");


# ### Question 4g
# In the context of this question, what are the benefit(s) of performing stratified sampling over cluster sampling? Why would you consider performing cluster sampling instead of stratified sampling? Compare the strengths and weaknesses of both sampling techniques.
# <!--
# BEGIN QUESTION
# name: q4g
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In stratified sampling, a random sample is drawn from each of the strata, whereas in cluster sampling only the selected clusters are sampled. 

# ### Question 4h
# Collect a multi-stage sample. First, take a SRS of 5 postal codes. Then, collect an SRS of one business name per selected postal code. Assign `bus_multi_sample` to a series of names selected by this procedure.
# 
# <!--
# BEGIN QUESTION
# name: q4h
# points: 1
# -->

# In[452]:


np.random.seed(40) # Do not touch this!
postal_codes = bus['postal_code_5'].unique()
sample_postal_codes = sample(postal_codes, 5)
sample_postal_codes
selected_rows = bus[bus['postal_code_5'].isin(sample_postal_codes)]
the_sample = selected_rows.groupby('postal_code_5').agg(sample, 1)
the_sample
the_sample['name']
# to get the values! use the function lambda z: z[0] to access
bus_multi_sample = the_sample['name'].agg(lambda z: z[0])
#bus_multi_sample = ...
bus_multi_sample.head()


# In[453]:


ok.grade("q4h");


# ### Question 4i
# What is the probability that AMERICANA GRILL & FOUNTAIN is chosen in the multi-stage sample?
# 
# <!--
# BEGIN QUESTION
# name: q4i
# points: 1
# -->

# In[130]:


part1 = 5/30
part2 = 1/160
q4i_answer = part1 * part2
q4i_answer


# In[131]:


ok.grade("q4i");


# ---
# ## 5: Latitude and Longitude
# 
# Let's also consider latitude and longitude values and get a sense of how many are missing.
# 
# ### Question 5a
# 
# How many businesses are missing longitude values?
# 
# *Hint*: Use `isnull`.
# 
# <!--
# BEGIN QUESTION
# name: q5a1
# points: 1
# -->

# In[455]:


#bus.head()
missing_longs = bus['longitude'].isnull()
num_missing_longs = sum(missing_longs)
num_missing_longs
#num_missing_longs = ...
#num_missing_longs


# In[456]:


ok.grade("q5a1");


# As a somewhat contrived exercise in data manipulation, let's try to identify which ZIP codes are missing the most longitude values.

# Throughout problems 5a and 5b, let's focus on only the "dense" ZIP codes of the city of San Francisco, listed below as `sf_dense_zip`.

# In[457]:


sf_dense_zip = ["94102", "94103", "94104", "94105", "94107", "94108",
                "94109", "94110", "94111", "94112", "94114", "94115",
                "94116", "94117", "94118", "94121", "94122", "94123", 
                "94124", "94127", "94131", "94132", "94133", "94134"]


# In the cell below, create a series where the index is `postal_code_5`, and the value is the number of businesses with missing longitudes in that ZIP code. Your series should be in descending order. Only businesses from `sf_dense_zip` should be included.
# 
# *Hint: Start by making a new dataframe called `bus_sf` that only has businesses from `sf_dense_zip`.*
# 
# *Hint: Create a custom function to compute the number of null entries in a series, and use this function with the `agg` method.*
# <!--
# BEGIN QUESTION
# name: q5a2
# points: 1
# -->

# In[458]:


dense_row = bus['postal_code_5'].isin(sf_dense_zip)
bus_sf = bus[dense_row]

def count_null_entries(dat) :
    null_row = dat['longitude'].isnull()
    nulls_dat = dat[null_row]
    return len(nulls_dat)


#bus_sf.groupby('postal_code_5').head()

x = bus_sf.groupby('postal_code_5').agg(count_null_entries)['business_id']
num_missing_in_each_zip = x.sort_values(ascending = False)
num_missing_in_each_zip.head()


# In[459]:


ok.grade("q5a2");


# ### Question 5b
# 
# In question 5a, we counted the number of null values per ZIP code. Let's now count the proportion of null values.
# 
# Create a new dataframe of counts of the null and proportion of null values, storing the result in `fraction_missing_df`. It should have an index called `postal_code_5` and should also have 3 columns:
# 
# 1. `count null`: The number of missing values for the zip code.
# 2. `count non null`: The number of present values for the zip code.
# 3. `fraction null`: The fraction of values that are null for the zip code.
# 
# Your data frame should be sorted by the fraction null in descending order.
# 
# Recommended approach: Build three series with the appropriate names and data and then combine them into a dataframe. This will require some new syntax you may not have seen. You already have code from question 4a that computes the `null count` series.
# 
# To pursue this recommended approach, you might find these two functions useful:
# 
# * `rename`: Renames the values of a series.
# * `pd.concat`: Can be used to combine a list of Series into a dataframe. Example: `pd.concat([s1, s2, s3], axis=1)` will combine series 1, 2, and 3 into a dataframe.
# 
# *Hint*: You can use the divison operator to compute the ratio of two series.
# 
# *Hint*: The - operator can invert a boolean array. Or alternately, the `notnull` method can be used to create a boolean array from a series.
# 
# *Note*: An alternate approach is to create three aggregation functions and pass them in a list to the `agg` function.
# <!--
# BEGIN QUESTION
# name: q5b
# points: 3
# -->

# In[460]:


#bus.groupby('postal_code_5').agg(count_null_entries)['business_id']

count_null = bus_sf[bus_sf['longitude'].isnull()].groupby('postal_code_5')['business_id'].count()
count_null.head()
count_non_null = bus_sf[bus_sf['longitude'].notnull()].groupby('postal_code_5')['business_id'].count()
count_non_null
fraction_null = bus_sf[bus_sf['longitude'].isnull()].groupby('postal_code_5')['business_id'].count() / bus_sf.groupby('postal_code_5')['business_id'].count()
fraction_null


fraction_missing_df = pd.concat([count_null.rename('count null'), count_non_null.rename('count non null'), fraction_null.rename('fraction null')], axis = 1)
fraction_missing_df.head()


# In[461]:


ok.grade("q5b");


# ## Summary of the Business Data
# 
# Before we move on to explore the other data, let's take stock of what we have learned and the implications of our findings on future analysis. 
# 
# * We found that the business id is unique across records and so we may be able to use it as a key in joining tables. 
# * We found that there are some errors with the ZIP codes. As a result, we dropped the records with ZIP codes outside of San Francisco or ones that were missing. In practive, however, we could take the time to look up the restaurant address online and fix these errors.   
# * We found that there are a huge number of missing longitude (and latitude) values. Fixing would require a lot of work, but could in principle be automated for records with well-formed addresses. 

# ---
# ## 6: Investigate the Inspection Data
# 
# Let's now turn to the inspection DataFrame. Earlier, we found that `ins` has 4 columns named `business_id`, `score`, `date` and `type`.  In this section, we determine the granularity of `ins` and investigate the kinds of information provided for the inspections. 

# Let's start by looking again at the first 5 rows of `ins` to see what we're working with.

# In[462]:


ins.head(5)


# ### Question 6a
# From calling `head`, we know that each row in this table corresponds to a single inspection. Let's get a sense of the total number of inspections conducted, as well as the total number of unique businesses that occur in the dataset.
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[463]:


# The number of rows in ins
rows_in_table  = len(ins)

# The number of unique business IDs in ins.
unique_ins_ids = len(ins['business_id'].unique())

ins['type'].unique()
ins['type'].value_counts()


# In[464]:


ok.grade("q6a");


# ### Question 6b
# 
# Next, we examine the Series in the `ins` dataframe called `type`. From examining the first few rows of `ins`, we see that `type` is a string and one of its values is `'routine'`, presumably for a routine inspection. What values does the inspection `type` take? How many occurrences of each value is in the DataFrame? What are the implications for further analysis? 
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The inspection type takes two different values: 'routine' and 'complaint'. The occurrence for 'routine' is 14221, and the occurrence for 'complaint' is 1. The implication can be that complaint might not have a significant affect on the result since there is only one record of it.

# ### Question 6c
# 
# In this question, we're going to try to figure out what years the data span. The dates in our file are formatted as strings such as `20160503`, which are a little tricky to interpret. The ideal solution for this problem is to modify our dates so that they are in an appropriate format for analysis. 
# 
# In the cell below, we attempt to add a new column to `ins` called `new_date` which contains the `date` stored as a datetime object. This calls the `pd.to_datetime` method, which converts a series of string representations of dates (and/or times) to a series containing a datetime object.

# In[465]:


ins['new_date'] = pd.to_datetime(ins['date'])
ins.head(5)


# As you'll see, the resulting `new_date` column doesn't make any sense. This is because the default behavior of the `to_datetime()` method does not properly process the passed string. We can fix this by telling `to_datetime` how to do its job by providing a format string.

# In[466]:


ins['new_date'] = pd.to_datetime(ins['date'], format='%Y%m%d')
ins.head(5)


# This is still not ideal for our analysis, so we'll add one more column that is just equal to the year by using the `dt.year` property of the new series we just created.

# In[467]:


ins['year'] = ins['new_date'].dt.year
ins.head(5)

ins['year'].sort_values()
ins['year'].unique()
ins['year'].value_counts().sort_values()


# Now that we have this handy `year` column, we can try to understand our data better.
# 
# What range of years is covered in this data set? Are there roughly the same number of inspections each year? Provide your answer in text only.
# 
# <!--
# BEGIN QUESTION
# name: q6c
# points: 1
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The range of years that is covered is from 2015 to 2018 (2016, 2017, 2015, 2018). There are not roughly the same number of inspections each year. For 2018, there are 308 records. For 2017, there are 5166 records. For 2016, there are 5443 records. For 2015, there are 3305 records. 

# ---
# ## 7: Explore Inspection Scores

# ### Question 7a
# Let's look at the distribution of inspection scores. As we saw before when we called `head` on this data frame, inspection scores appear to be integer values. The discreteness of this variable means that we can use a barplot to visualize the distribution of the inspection score. Make a bar plot of the counts of the number of inspections receiving each score. 
# 
# It should look like the image below. It does not need to look exactly the same, but make sure that all labels and axes are correct.
# 
# <img src="q7a.png" width=500>
# 
# <!--
# BEGIN QUESTION
# name: q7a
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[469]:


#...
inspect_hist = sns.distplot(ins['score'], kde = False)
inspect_hist.set_xlabel('Score')
inspect_hist.set_ylabel('Count')
inspect_hist.set_title('Distribution of Inspection Scores')


# ### Question 7b
# 
# Describe the qualities of the distribution of the inspections scores based on your bar plot. Consider the mode(s), symmetry, tails, gaps, and anamolous values. Are there any unusual features of this distribution? What do your observations imply about the scores?
# 
# <!--
# BEGIN QUESTION
# name: q7b
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# We can see that the histogram is skewed to the left, where the left tail is pretty long. This tells us, there's a lot of inspections that got a high score. The plot also tells us that there is no symmetry. Moreover, there is a a big gap from 0 to 65. This implies that there is barely low scores (under 65). The observation implies most of the scores are in 90 to 100 range. 

# ### Question 7c

# Let's figure out which restaurants had the worst scores ever. Let's start by creating a new dataframe called `ins_named`. It should be exactly the same as `ins`, except that it should have the name and address of every business, as determined by the `bus` dataframe. If a `business_id` in `ins` does not exist in `bus`, the name and address should be given as NaN.
# 
# *Hint: Use the merge method to join the `ins` dataframe with the appropriate portion of the `bus` dataframe.*
# 
# <!--
# BEGIN QUESTION
# name: q7c1
# points: 1
# -->

# In[470]:


merge_whole = ins.merge(bus, how = 'left')
merge_whole
ins_named = merge_whole.loc[:,'business_id':'address']

#ins_named = ...
ins_named.head()

sorted_score = ins_named.groupby('score')
#sorted_score.head().sort_values('score')


# In[471]:


ok.grade("q7c1");


# Using this data frame, identify the restaurant with the lowest inspection scores ever. Head to yelp.com and look up the reviews page for this restaurant. Copy and paste anything interesting you want to share.
# 
# <!--
# BEGIN QUESTION
# name: q7c2
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# The lowest inspection scores belongs to DA Cafe which is located at 407 CLEMENT ST. It has the score of 48.

# Just for fun you can also look up the restaurants with the best scores. You'll see that lots of them aren't restaurants at all!

# ---
# ## 8: Restaurant Ratings Over Time

# Let's consider various scenarios involving restaurants with multiple ratings over time.

# ### Question 8a

# Let's see which restaurant has had the most extreme improvement in its rating. Let the "swing" of a restaurant be defined as the difference between its highest and lowest rating ever. **Only consider restaurants with at least 3 ratings!** Using whatever technique you want to use, assign `max_swing` to the name of restaurant that has the maximum swing.
# 
# <!--
# BEGIN QUESTION
# name: q8a1
# points: 2
# -->

# In[472]:


ins_named.head()
#ins_named.groupby('business_id')

at_least_3 = ins_named.groupby('business_id').count()[ins_named.groupby('business_id')['score'].count() >= 3]
al3 = ins_named[ins_named['business_id'].isin(list(at_least_3.index.values))]

min_score = al3.groupby('business_id').agg(min)['score']
max_score = al3.groupby('business_id').agg(max)['score']

bind = pd.concat([min_score.rename('min'),max_score.rename('max')],axis = 1)
bind.head()
bind['diff'] = bind['max'] - bind['min']
the_big_imp_id = bind['diff'].sort_values(ascending = False).index[0]
max_swing = list(bus.loc[bus['business_id'] == the_big_imp_id, 'name'])[0]
max_swing


# In[473]:


ok.grade("q8a1");


# ### Question 8b
# 
# To get a sense of the number of times each restaurant has been inspected, create a multi-indexed dataframe called `inspections_by_id_and_year` where each row corresponds to data about a given business in a single year, and there is a single data column named `count` that represents the number of inspections for that business in that year. The first index in the MultiIndex should be on `business_id`, and the second should be on `year`.
# 
# An example row in this dataframe might look tell you that business_id is 573, year is 2017, and count is 4.
# 
# *Hint: Use groupby to group based on both the `business_id` and the `year`.*
# 
# *Hint: Use rename to change the name of the column to `count`.*
# 
# <!--
# BEGIN QUESTION
# name: q8b
# points: 2
# -->

# In[474]:


ins_named.head()
grouped = ins_named.groupby(['business_id', 'year']).size()
inspections_by_id_and_year = grouped.sort_values(ascending=False).rename('count').to_frame()


# In[475]:


ok.grade("q8b");


# You should see that some businesses are inspected many times in a single year. Let's get a sense of the distribution of the counts of the number of inspections by calling `value_counts`. There are quite a lot of businesses with 2 inspections in the same year, so it seems like it might be interesting to see what we can learn from such businesses.

# In[476]:


inspections_by_id_and_year['count'].value_counts()


# ### Question 8c
# 
# What's the relationship between the first and second scores for the businesses with 2 inspections in a year? Do they typically improve? For simplicity, let's focus on only 2016 for this problem.
# 
# First, make a dataframe called `scores_pairs_by_business` indexed by `business_id` (containing only businesses with exactly 2 inspections in 2016).  This dataframe contains the field `score_pair` consisting of the score pairs ordered chronologically  `[first_score, second_score]`. 
# 
# Plot these scores. That is, make a scatter plot to display these pairs of scores. Include on the plot a reference line with slope 1. 
# 
# You may find the functions `sort_values`, `groupby`, `filter` and `agg` helpful, though not all necessary. 
# 
# The first few rows of the resulting table should look something like:
# 
# <table border="1" class="dataframe">
#   <thead>
#     <tr style="text-align: right;">
#       <th></th>
#       <th>score_pair</th>
#     </tr>
#     <tr>
#       <th>business_id</th>
#       <th></th>
#     </tr>
#   </thead>
#   <tbody>
#     <tr>
#       <th>24</th>
#       <td>[96, 98]</td>
#     </tr>
#     <tr>
#       <th>45</th>
#       <td>[78, 84]</td>
#     </tr>
#     <tr>
#       <th>66</th>
#       <td>[98, 100]</td>
#     </tr>
#     <tr>
#       <th>67</th>
#       <td>[87, 94]</td>
#     </tr>
#     <tr>
#       <th>76</th>
#       <td>[100, 98]</td>
#     </tr>
#   </tbody>
# </table>
# 
# The scatter plot should look like this:
# 
# <img src="q8c2.png" width=500>
# 
# *Note: Each score pair must be a list type; numpy arrays will not pass the autograder.*
# 
# *Hint: Use the `filter` method from lecture 3 to create a new dataframe that only contains restaurants that received exactly 2 inspections.*
# 
# *Hint: Our answer is a single line of code that uses `sort_values`, `groupby`, `filter`, `groupby`, `agg`, and `rename` in that order. Your answer does not need to use these exact methods.*
# 
# <!--
# BEGIN QUESTION
# name: q8c1
# points: 3
# -->

# In[477]:


# Create the dataframe here

def turnlist(x):
    return list(x)

ins2016 = ins[ins['year'] == 2016]
scores_pairs_by_business = ins2016.sort_values('date').loc[:, 'business_id':'score'].groupby('business_id').filter(lambda x: len(x) == 2).groupby('business_id').agg(turnlist).rename(columns = {'score': 'score_pair'})





# In[478]:


ok.grade("q8c1");


# Now, create your scatter plot in the cell below.
# <!--
# BEGIN QUESTION
# name: q8c2
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[479]:


#...
x, y = zip(*scores_pairs_by_business['score_pair'])

#sns.lmplot(x='First Score', y='Second Score', data=scores_pairs_by_business['score_pair'])

plt.scatter(x, y, facecolors = 'none', edgecolors='b')
plt.plot([55,100],[55,100],'r-')
plt.axis([55,100,55,100]);
plt.xlabel('first score')
plt.ylabel('second score')
plt.title('First Inspection Score vs. Second Inspection Score')
plt.grid(True)


# ### Question 8d
# 
# Another way to compare the scores from the two inspections is to examine the difference in scores. Subtract the first score from the second in `scores_pairs_by_business`. Make a histogram of these differences in the scores. We might expect these differences to be positive, indicating an improvement from the first to the second inspection.
# 
# The histogram should look like this:
# 
# <img src="q8d.png" width=500>
# 
# *Hint: Use `second_score` and `first_score` created in the scatter plot code above.*
# 
# *Hint: Convert the scores into numpy arrays to make them easier to deal with.*
# 
# *Hint: Try changing the number of bins when you call plt.hist.*
# 
# <!--
# BEGIN QUESTION
# name: q8d
# points: 2
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In[480]:


#...
arr_x = np.array(x)
arr_y = np.array(y)
y_x = arr_y - arr_x
plt.hist(y_x, bins = 30)
plt.grid(True)
plt.title('Distribution of Score Differences')
plt.ylabel('Count')
plt.xlabel('Score Difference (Second Score - First Score)')


# ### Question 8e
# 
# If a restaurant's score improves from the first to the second inspection, what do you expect to see in the scatter plot that you made in question 8c? What do you see?
# 
# If a restaurant's score improves from the first to the second inspection, how would this be reflected in the histogram of the difference in the scores that you made in question 8d? What do you see?
# 
# <!--
# BEGIN QUESTION
# name: q8e
# points: 3
# manual: True
# -->
# <!-- EXPORT TO PDF -->

# In the scatter plot, the dots will tend to move up.
# In the histogram, values will tend to move right.

# ## Summary of the Inspections Data
# 
# What we have learned about the inspections data? What might be some next steps in our investigation? 
# 
# * We found that the records are at the inspection level and that we have inspections for multiple years.   
# * We also found that many restaurants have more than one inspection a year. 
# * By joining the business and inspection data, we identified the name of the restaurant with the worst rating and optionally the names of the restaurants with the best rating.
# * We identified the restaurant that had the largest swing in rating over time.
# * We also examined the relationship between the scores when a restaurant has multiple inspections in a year. Our findings were a bit counterintuitive and may warrant further investigation. 
# 

# ## Congratulations!
# 
# You are finished with Project 1. You'll need to make sure that your PDF exports correctly to receive credit. Run the following cell and follow the instructions.

# In[481]:


# Save your notebook first, then run this cell to submit.
import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj1.ipynb', 'proj1.pdf')
ok.submit()


# In[ ]:




