#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Initialize OK
from client.api.notebook import Notebook
ok = Notebook('proj2.ok')


# # Project : Spam/Ham Classification
# ## Feature Engineering, Logistic Regression, Cross Validation
# 


# ## This Project
# In this project, create a classifier that can distinguish spam (junk or commercial or bulk) emails from ham (non-spam) emails.
# 
# 
# - Feature engineering with text data
# - Using sklearn libraries to process data and fit models
# - Validating the performance of your model and minimizing overfitting
# - Generating and analyzing precision-recall curves
# 

# # Part I - Initial Analysis

# In[2]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import seaborn as sns
sns.set(style = "whitegrid", 
        color_codes = True,
        font_scale = 1.5)


# ### Loading in the Data
# 
# In email classification, our goal is to classify emails as spam or not spam (referred to as "ham") using features generated from the text in the email. 
# 
# The dataset consists of email messages and their labels (0 for ham, 1 for spam). The labeled training dataset contains 8348 labeled examples, and the test set contains 1000 unlabeled examples.
# 
# Run the following cells to load in the data into DataFrames.
# 
# The `train` DataFrame contains labeled data that will be used to train your model. It contains four columns:
# 
# 1. `id`: An identifier for the training example
# 1. `subject`: The subject of the email
# 1. `email`: The text of the email
# 1. `spam`: 1 if the email is spam, 0 if the email is ham (not spam)
# 
# The `test` DataFrame contains 1000 unlabeled emails. Predict labels for these emails.

# In[3]:


from utils import fetch_and_cache_gdrive
fetch_and_cache_gdrive('1SCASpLZFKCp2zek-toR3xeKX3DZnBSyp', 'train.csv')
fetch_and_cache_gdrive('1ZDFo9OTF96B5GP2Nzn8P8-AL7CTQXmC0', 'test.csv')

original_training_data = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

# Convert the emails to lower case as a first step to processing the text
original_training_data['email'] = original_training_data['email'].str.lower()
test['email'] = test['email'].str.lower()

original_training_data.head()


# ### Question 1a
# First, let's check if our data contains any missing values. Fill in the cell below to print the number of NaN values in each column. If there are NaN values, replace them with appropriate filler values (i.e., NaN values in the `subject` or `email` columns should be replaced with empty strings). Print the number of NaN values in each column after this modification to verify that there are no NaN values left.
# 
# Note that while there are no NaN values in the `spam` column, we should be careful when replacing NaN labels. Doing so without consideration may introduce significant bias into our model when fitting.
# 
# *The provided test checks that there are no missing values in your dataset.*
# 
# <!--
# BEGIN QUESTION
# name: q1a
# points: 1
# -->

# In[4]:


# ...
# train = 
# train.head()
original_training_data['subject'][original_training_data['subject'].isnull()]
original_training_data.isnull().agg(sum)

original_training_data['subject'][original_training_data['subject'].isnull()] = ""
original_training_data.isnull().agg(sum)


# In[5]:


ok.grade("q1a");


# ### Question 1b
# 
# In the cell below, print the text of the first ham and the first spam email in the original training set.
# 
# *The provided tests just ensure that you have assigned `first_ham` and `first_spam` to rows in the data, but only the hidden tests check that you selected the correct observations.*
# 
# <!--
# BEGIN QUESTION
# name: q1b
# points: 1
# -->

# In[6]:


# spam: 1 if the email is spam, 0 if the email is ham (not spam)

first_ham = original_training_data[original_training_data['spam'] == 0].head(1)['subject'].values[0]
first_spam = original_training_data[original_training_data['spam'] == 1].head(1)['subject'].values[0]
print(first_ham)
print(first_spam)
# type(first_ham)
# first_spam
# first_ham


# In[7]:


ok.grade("q1b");


# ### Question 1c
# 
# Discuss one thing you notice that is different between the two emails that might relate to the identification of spam.
# 
# <!--
# BEGIN QUESTION
# name: q1c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# Spam mail is more subjective, whereas ham mail is more objective. 

# ## Training Validation Split
# The training data we downloaded is all the data we have available for both training models and **validating** the models that we train.  We therefore need to split the training data into separate training and validation datsets.  You will need this **validation data** to assess the performance of your classifier once you are finished training. Note that we set the seed (random_state) to 42. This will produce a pseudo-random sequence of random numbers that is the same for every student. Do not modify this in the following questions, as our tests depend on this random seed.

# In[8]:


from sklearn.model_selection import train_test_split

train, val = train_test_split(original_training_data, test_size=0.1, random_state=42)


# # Basic Feature Engineering
# 
# We would like to take the text of an email and predict whether the email is ham or spam. This is a *classification* problem, so we can use logistic regression to train a classifier. Recall that to train an logistic regression model we need a numeric feature matrix $X$ and a vector of corresponding binary labels $y$.  Unfortunately, our data are text, not numbers. To address this, we can create numeric features derived from the email text and use those features for logistic regression.
# 
# Each row of $X$ is an email. Each column of $X$ contains one feature for all the emails. We'll guide you through creating a simple feature, and you'll create more interesting ones when you are trying to increase your accuracy.

# ### Question 2
# 
# Create a function called `words_in_texts` that takes in a list of `words` and a pandas Series of email `texts`. It should output a 2-dimensional NumPy array containing one row for each email text. The row should contain either a 0 or a 1 for each word in the list: 0 if the word doesn't appear in the text and 1 if the word does. For example:
# 
# ```
# >>> words_in_texts(['hello', 'bye', 'world'], 
#                    pd.Series(['hello', 'hello worldhello']))
# 
# array([[1, 0, 0],
#        [1, 0, 1]])
# ```
# 
# *The provided tests make sure that your function works correctly, so that you can use it for future questions.*
# 
# <!--
# BEGIN QUESTION
# name: q2
# points: 3
# -->

# In[9]:


def words_in_texts(words, texts):
    '''
    Args:
        words (list-like): words to find
        texts (Series): strings to search in
    
    Returns:
        NumPy array of 0s and 1s with shape (n, p) where n is the
        number of texts and p is the number of words.
    '''
#     indicator_array = ...
    
    indicator_array = np.array([texts.str.contains(word) for word in words]).astype(int).T

    return indicator_array

words_in_texts(['hello', 'bye', 'world'], 
                   pd.Series(['hello', 'hello worldhello']))


# In[10]:


ok.grade("q2");


# # Basic EDA
# 
# We need to identify some features that allow us to distinguish spam emails from ham emails. One idea is to compare the distribution of a single feature in spam emails to the distribution of the same feature in ham emails. If the feature is itself a binary indicator, such as whether a certain word occurs in the text, this amounts to comparing the proportion of spam emails with the word to the proportion of ham emails with the word.
# 

# The following plot (which was created using `sns.barplot`) compares the proportion of emails in each class containing a particular set of words. 
# 
# ![training conditional proportions](./images/training_conditional_proportions.png "Class Conditional Proportions")
# 
# Hint:
# - You can use DataFrame's `.melt` method to "unpivot" a DataFrame. See the following code cell for an example.

# In[11]:


from IPython.display import display, Markdown
df = pd.DataFrame({
    'word_1': [1, 0, 1, 0],
    'word_2': [0, 1, 0, 1],
    'type': ['spam', 'ham', 'ham', 'ham']
})
display(Markdown("> Our Original DataFrame has some words column and a type column. You can think of each row is a sentence, and the value of 1 or 0 indicates the number of occurances of the word in this sentence."))
display(df);
display(Markdown("> `melt` will turn columns into variale, notice how `word_1` and `word_2` become `variable`, their values are stoed in the value column"))
display(df.melt("type"))


# ### Question 3a
# 
# Create a bar chart like the one above comparing the proportion of spam and ham emails containing certain words. Choose a set of words that are different from the ones above, but also have different proportions for the two classes. Make sure to only consider emails from `train`.
# 
# <!--
# BEGIN QUESTION
# name: q3a
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[12]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts

words = ['http', 'sex', '<', 'please']
# words = ['drug', 'bank', 'prescription', 'memo', 'private']
spams = train[train['spam'] == 1]
hams = train[train['spam'] == 0]
spams_text = words_in_texts(words, spams['email'])
hams_text = words_in_texts(words, hams['email'])

spams_prop = [sum(col)/len(col) for col in spams_text.T]
hams_prop = [sum(col)/len(col) for col in hams_text.T]


df = pd.DataFrame()
df["prop"] = np.append(hams_prop, spams_prop)
df["type"] = np.append(np.repeat('Ham', len(words)), np.repeat('Spam', len(words)))
df["words"] =  words + words
df.head()

sns.barplot(data = df, x = "words", y = "prop", hue="type")
plt.ylim(0,1)


# When the feature is binary, it makes sense to compare its proportions across classes (as in the previous question). Otherwise, if the feature can take on numeric values, we can compare the distributions of these values for different classes. 
# 
# ![training conditional densities](./images/training_conditional_densities2.png "Class Conditional Densities")
# 

# ### Question 3b
# 
# Create a *class conditional density plot* like the one above (using `sns.distplot`), comparing the distribution of the length of spam emails to the distribution of the length of ham emails in the training set. Set the x-axis limit from 0 to 50000.
# 
# <!--
# BEGIN QUESTION
# name: q3b
# manual: True
# format: image
# points: 2
# -->
# <!-- EXPORT TO PDF format:image -->

# In[13]:


# comparing the distribution of the length of spam emails vs ham emails

# ...
ham_len = hams['email'].str.len()
spam_len = spams['email'].str.len()

# hams.head()
# ham_len.head(10)

sns.distplot(ham_len, label = 'Ham', hist = False)
sns.distplot(spam_len, label = 'Spam', hist = False)
plt.xlim(0, 50000)
plt.legend()
plt.xlabel("Length of email body")
plt.ylabel("Distribution")


# # Basic Classification
# 
# Notice that the output of `words_in_texts(words, train['email'])` is a numeric matrix containing features for each email. This means we can use it directly to train a classifier!

# ### Question 4
# 
# We've given you 5 words that might be useful as features to distinguish spam/ham emails. Use these words as well as the `train` DataFrame to create two NumPy arrays: `X_train` and `Y_train`.
# 
# `X_train` should be a matrix of 0s and 1s created by using your `words_in_texts` function on all the emails in the training set.
# 
# `Y_train` should be a vector of the correct labels for each email in the training set.
# 
# *The provided tests check that the dimensions of your feature matrix (X) are correct, and that your features and labels are binary (i.e. consists of 0 and 1, no other values). It does not check that your function is correct; that was verified in a previous question.*
# <!--
# BEGIN QUESTION
# name: q4
# points: 2
# -->

# In[14]:


some_words = ['drug', 'bank', 'prescription', 'memo', 'private']

X_train = words_in_texts(some_words, train['email'])
Y_train = train['spam']

X_train[:5], Y_train[:5]
X_train


# In[15]:


ok.grade("q4");


# ### Question 5
# 
# Now we have matrices we can give to scikit-learn! Using the [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) classifier, train a logistic regression model using `X_train` and `Y_train`. Then, output the accuracy of the model (on the training data) in the cell below. You should get an accuracy around 0.75.
# 
# *The provided test checks that you initialized your logistic regression model correctly.*
# 
# <!--
# BEGIN QUESTION
# name: q5
# points: 2
# -->

# In[16]:


from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, Y_train)
# ...

y_fitted = model.predict(X_train)

training_accuracy = model.score(X_train, Y_train)
# training_accuracy = ...
print("Training Accuracy: ", training_accuracy)


# In[17]:


ok.grade("q5");


# ## Evaluating Classifiers

# That doesn't seem too shabby! But the classifier you made above isn't as good as this might lead us to believe. First, we are evaluating accuracy on the training set, which may lead to a misleading accuracy measure, especially if we used the training set to identify discriminative features. In future parts of this analysis, it will be safer to hold out some of our data for model validation and comparison.
# 
# Presumably, our classifier will be used for **filtering**, i.e. preventing messages labeled `spam` from reaching someone's inbox. There are two kinds of errors we can make:
# - False positive (FP): a ham email gets flagged as spam and filtered out of the inbox.
# - False negative (FN): a spam email gets mislabeled as ham and ends up in the inbox.
# 
# These definitions depend both on the true labels and the predicted labels. False positives and false negatives may be of differing importance, leading us to consider more ways of evaluating a classifier, in addition to overall accuracy:
# 
# **Precision** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FP}}$ of emails flagged as spam that are actually spam.
# 
# **Recall** measures the proportion $\frac{\text{TP}}{\text{TP} + \text{FN}}$ of spam emails that were correctly flagged as spam. 
# 
# **False-alarm rate** measures the proportion $\frac{\text{FP}}{\text{FP} + \text{TN}}$ of ham emails that were incorrectly flagged as spam. 
# 
# The following image might help:
# 
# <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Precisionrecall.svg/700px-Precisionrecall.svg.png" width="500px">
# 
# Note that a true positive (TP) is a spam email that is classified as spam, and a true negative (TN) is a ham email that is classified as ham.

# ### Question 6a
# 
# Suppose we have a classifier `zero_predictor` that always predicts 0 (never predicts positive). How many false positives and false negatives would this classifier have if it were evaluated on the training set and its results were compared to `Y_train`? Fill in the variables below (answers can be hard-coded):
# 
# *Tests in Question 6 only check that you have assigned appropriate types of values to each response variable, but do not check that your answers are correct.*
# 
# <!--
# BEGIN QUESTION
# name: q6a
# points: 1
# -->

# In[18]:


zero_predictor_fp = 0
zero_predictor_fn = len(train[train['spam'] == 1])


# In[19]:


ok.grade("q6a");


# ### Question 6b
# 
# What are the accuracy and recall of `zero_predictor` (classifies every email as ham) on the training set? Do NOT use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6b
# points: 1
# -->

# In[20]:


zero_predictor_acc = (1/len(X_train)) * len(train[train['spam'] == 0])
zero_predictor_recall = 0
zero_predictor_acc
# zero_predictor_recall


# In[21]:


ok.grade("q6b");


# ### Question 6c
# 
# Provide brief explanations of the results from 6a and 6b. Why do we observe each of these values (FP, FN, accuracy, recall)?
# 
# <!--
# BEGIN QUESTION
# name: q6c
# manual: True
# points: 2
# -->
# <!-- EXPORT TO PDF -->

# *Write your answer here, replacing this text.*

# ### Question 6d
# 
# Compute the precision, recall, and false-alarm rate of the `LogisticRegression` classifier created and trained in Question 5. Do NOT use any `sklearn` functions.
# 
# <!--
# BEGIN QUESTION
# name: q6d
# points: 2
# -->

# In[23]:


tp = np.count_nonzero((y_fitted == 1) & (y_fitted == Y_train))
tn = np.count_nonzero((y_fitted == 0)& (y_fitted == Y_train))
fp = np.count_nonzero((y_fitted == 1) & (y_fitted != Y_train))
fn = np.count_nonzero((y_fitted == 0) & (y_fitted != Y_train))


# logistic_predictor_precision = ...
logistic_predictor_precision = tp/(tp+fp)
logistic_predictor_recall = tp/(tp+fn)
logistic_predictor_far = fp/(fp+tn)


# In[51]:


ok.grade("q6d");


# ### Question 6e
# 
# Are there more false positives or false negatives when using the logistic regression classifier from Question 5?
# 
# <!--
# BEGIN QUESTION
# name: q6e
# manual: True
# points: 1
# -->
# <!-- EXPORT TO PDF -->

# More false positive and less false negative.

# ### Question 6f
# 
# 1. Our logistic regression classifier got 75.6% prediction accuracy (number of correct predictions / total). How does this compare with predicting 0 for every email?
# 1. Given the word features we gave you above, name one reason this classifier is performing poorly. Hint: Think about how prevalent these words are in the email set.
# 1. Which of these two classifiers would you prefer for a spam filter and why? Describe your reasoning and relate it to at least one of the evaluation metrics you have computed so far.
# 
# <!--
# BEGIN QUESTION
# name: q6f
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# 1. The 75.6% accuracy is way better than the 0% accuracy of the always 0 predictor.
# 2. There is no email being clasified as spam which is kind of unlikely in the real world. This might also mean that the words we choose to consider in making our model isn't a very good option. It seems that these words appears in both spam and ham emails. 
# 3. I prefer logistic because it has higher accuracy and is more realistic.
# 

# # Part II - Moving Forward
# 
# With this in mind, it is now your task to make the spam filter more accurate. In order to get full credit on the accuracy part of this assignment, you must get at least **88%** accuracy on the test set. To see your accuracy on the test set, you will use your classifier to predict every email in the `test` DataFrame and upload your predictions to Kaggle.
# 
# **Kaggle limits you to four submissions per day**. This means you should start early so you have time if needed to refine your model. You will be able to see your accuracy on the entire set when submitting to Kaggle (the accuracy that will determine your score for question 10).
# 
# Here are some ideas for improving your model:
# 
# 1. Finding better features based on the email text. Some example features are:
#     1. Number of characters in the subject / body
#     1. Number of words in the subject / body
#     1. Use of punctuation (e.g., how many '!' were there?)
#     1. Number / percentage of capital letters 
#     1. Whether the email is a reply to an earlier email or a forwarded email
# 1. Finding better words to use as features. Which words are the best at distinguishing emails? This requires digging into the email text itself. 
# 1. Better data processing. For example, many emails contain HTML as well as text. You can consider extracting out the text from the HTML to help you find better words. Or, you can match HTML tags themselves, or even some combination of the two.
# 1. Model selection. You can adjust parameters of your model (e.g. the regularization parameter) to achieve higher accuracy. Recall that you should use cross-validation to do feature and model selection properly! Otherwise, you will likely overfit to your training data.
# 
# ou may use whatever method you prefer in order to create features, but **you are not allowed to import any external feature extraction libraries**. In addition, **you are only allowed to train logistic regression models**. No random forests, k-nearest-neighbors, neural nets, etc.
# 
# We have not provided any code to do this, so feel free to create as many cells as you need in order to tackle this task. However, answering questions 7, 8, and 9 should help guide you.
# 
# ---
# 
# **Note:** *You should use the **validation data** to evaluate your model and get a better sense of how it will perform on the Kaggle evaluation.*
# 
# ---

# In[52]:


train=train.reset_index(drop=True) # We must do this in order to preserve the ordering of emails to labels for words_in_texts

words = ['free', 'please', 'news', '%', '#', 'offer', '!',
              'face', 'size', 'width', 'font', 'td', 
              'promotion', '!!', '50%', '30%', 'sample',
              'sale', '##', 'click']
spams = train[train['spam'] == 1]
hams = train[train['spam'] == 0]
spams_text = words_in_texts(words, spams['email'])
hams_text = words_in_texts(words, hams['email'])

spams_prop = [sum(col)/len(col) for col in spams_text.T]
hams_prop = [sum(col)/len(col) for col in hams_text.T]


df = pd.DataFrame()
df["prop"] = np.append(hams_prop, spams_prop)
df["type"] = np.append(np.repeat('Ham', len(words)), np.repeat('Spam', len(words)))
df["words"] =  words + words
df.head()

sns.barplot(data = df, x = "words", y = "prop", hue="type")
plt.ylim(0,1)


# In[53]:


some_words = ['free', 'please', 'news', '%', '#', 'offer', '!',
              'face', 'size', 'width', 'font', 'td', 
              'promotion', '!!', '50%', '30%', 'sample',
              'sale', '##', 'click']

xtrain = words_in_texts(some_words, train['email'])
ytrain = train['spam']

xtrain[:5]


# In[54]:


lr = LogisticRegression()
lr.fit(xtrain, ytrain)
# ...

y_fitted = lr.predict(xtrain)

training_accuracy = lr.score(xtrain, ytrain)
# training_accuracy = ...
print("Training Accuracy: ", training_accuracy)


# ### Question 7: Feature/Model Selection Process
# 
# In this following cell, describe the process of improving your model. You should use at least 2-3 sentences each to address the follow questions:
# 
# 1. How did you find better features for your model?
# 2. What did you try that worked / didn't work?
# 3. What was surprising in your search for good features?
# 
# <!--
# BEGIN QUESTION
# name: q7
# manual: True
# points: 6
# -->
# <!-- EXPORT TO PDF -->

# 1. First I used worldcloud to find the interesting words. After that, I tried several different combination of words that might distinguish spam and ham, and use the bar chart to see if the difference between the two is big.
# 2. I also tried to use length to distinguish, however, the difference isn't very significant.
# 3. Existense of punctuation seems to be a good feature that helps improving model.

# ### Question 8: EDA
# 
# In the cell below, show a visualization that you used to select features for your model. Include both
# 
# 1. A plot showing something meaningful about the data that helped you during feature / model selection.
# 2. 2-3 sentences describing what you plotted and what its implications are for your features.
# 
# Feel to create as many plots as you want in your process of feature selection, but select one for the response cell below.
# 
# **You should not just produce an identical visualization to question 3.** Specifically, don't show us a bar chart of proportions, or a one-dimensional class-conditional density plot. Any other plot is acceptable, as long as it comes with thoughtful commentary. Here are some ideas:
# 
# 1. Consider the correlation between multiple features (look up correlation plots and `sns.heatmap`). 
# 1. Try to show redundancy in a group of features (e.g. `body` and `html` might co-occur relatively frequently, or you might be able to design a feature that captures all html tags and compare it to these). 
# 1. Visualize which words have high or low values for some useful statistic.
# 1. Visually depict whether spam emails tend to be wordier (in some sense) than ham emails.

# Generate your visualization in the cell below and provide your description in a comment.
# 
# <!--
# BEGIN QUESTION
# name: q8
# manual: True
# format: image
# points: 6
# -->
# <!-- EXPORT TO PDF format:image -->

# In[55]:


# Write your description (2-3 sentences) as a comment here:
# I used wordcloud to find some interesting words that can distinguish the two groups: spam and ham emails.
# From the output, we can see that the words that has the bigger font (font, size, face) represent 
# the significant words/features that can help us improve the accuracy of our model.
#
#

# Write the code to generate your visualization here:
get_ipython().system('pip install wordcloud')
from wordcloud import WordCloud
spamWords = ' '.join(list(train[train['spam'] == 1]['email']))
cloud = WordCloud(width = 1024,height = 1024).generate(spamWords)
plt.imshow(cloud)
plt.axis('off')
plt.show()


# ### Question 9: Precision-Recall Curve
# 
# We can trade off between precision and recall. In most cases we won't be able to get both perfect precision (i.e. no false positives) and recall (i.e. no false negatives), so we have to compromise. For example, in the case of cancer screenings, false negatives are comparatively worse than false positives — a false negative means that a patient might not discover a disease until it's too late to treat, while a false positive means that a patient will probably have to take another screening.
# 
# Recall that logistic regression calculates the probability that an example belongs to a certain class. Then, to classify an example we say that an email is spam if our classifier gives it $\ge 0.5$ probability of being spam. However, *we can adjust that cutoff*: we can say that an email is spam only if our classifier gives it $\ge 0.7$ probability of being spam, for example. This is how we can trade off false positives and false negatives.
# 
# The precision-recall curve shows this trade off for each possible cutoff probability. In the cell below, [plot a precision-recall curve](http://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html#plot-the-precision-recall-curve) for your final classifier (the one you use to make predictions for Kaggle).
# 
# <!--
# BEGIN QUESTION
# name: q9
# manual: True
# points: 3
# -->
# <!-- EXPORT TO PDF -->

# In[56]:


from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
# Note that you'll want to use the .predict_proba(...) method for your classifier
# instead of .predict(...) so you get probabilities, not classes
# ...
y_score = lr.decision_function(xtrain)
average_precision = average_precision_score(ytrain, y_score)

precision, recall, _ = precision_recall_curve(ytrain, y_score)

plt.step(recall, precision, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
# ...


# # Question 10: Submitting to Kaggle
# 
# The following code will write your predictions on the test dataset to a CSV, which you can submit to Kaggle. You may need to modify it to suit your needs.
# 
# Save your predictions in a 1-dimensional array called `test_predictions`. *Even if you are not submitting to Kaggle, please make sure you've saved your predictions to `test_predictions` as this is how your score for this question will be determined.*
# 
# Remember that if you've performed transformations or featurization on the training data, you must also perform the same transformations on the test data in order to make predictions. For example, if you've created features for the words "drug" and "money" on the training data, you must also extract the same features in order to use scikit-learn's `.predict(...)` method.
# 
# You should submit your CSV files to https://www.kaggle.com/t/1b01c36920b74f0e8502a69760775a1d
# 
# *The provided tests check that your predictions are in the correct format, but you must submit to Kaggle to evaluate your classifier accuracy.*
# 
# <!--
# BEGIN QUESTION
# name: q10
# points: 15
# -->

# In[57]:


# test_predictions = ...

words_in_texts(some_words, test['email'])

y_predict = lr.predict(words_in_texts(some_words, test['email']))
test_predictions = y_predict


# In[58]:


ok.grade("q10");


# The following saves a file to submit to Kaggle.

# In[272]:


from datetime import datetime

# Assuming that your predictions on the test set are stored in a 1-dimensional array called
# test_predictions. Feel free to modify this cell as long you create a CSV in the right format.

# Construct and save the submission:
submission_df = pd.DataFrame({
    "Id": test['id'], 
    "Class": test_predictions,
}, columns=['Id', 'Class'])
timestamp = datetime.isoformat(datetime.now()).split(".")[0]
submission_df.to_csv("submission_{}.csv".format(timestamp), index=False)

print('Created a CSV file: {}.'.format("submission_{}.csv".format(timestamp)))
print('You may now upload this CSV file to Kaggle for scoring.')



# 
# <!-- EXPECT 9 EXPORTED QUESTIONS -->

# In[59]:


import jassign.to_pdf
jassign.to_pdf.generate_pdf('proj2.ipynb', 'proj2.pdf')
ok.submit()


# In[ ]:




