# Sentiment Analysis
# ----------------------------------------------
# Author: Adrien Talbot (MSc Business Analytics, Imperial College London)
# Twitter-API documentation: https://developer.twitter.com/en/docs/tweets/search/api-reference/get-search-tweets
# News-API documentation: https://newsapi.org/docs/endpoints/everything

# Improvements:
#   - Obtain News API pro account so that the news content older than 1 month can be extracted (especially for time series analysis)

import twitter
import pandas as pd
import re
from nltk.tokenize import word_tokenize
from string import punctuation
from nltk.corpus import stopwords
import nltk
import requests
import numpy as np
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from newsapi import NewsApiClient
import datetime
from dateutil.relativedelta import relativedelta
import dateutil
from newsapi import NewsApiClient
import datetime
import matplotlib.pyplot as plt
import numpy as np

# nltk.download('popular') #We need this if certain nltk libraries are not installed.


# create the test set for tweets:
def twitter_test_set(string_buzzword, consumer_key, consumer_secret, access_token_key, access_token_secret):

    # Authenticating our twitter API credentials
    twitter_api = twitter.Api(consumer_key=consumer_key,
                              consumer_secret=consumer_secret,
                              access_token_key=access_token_key,
                              access_token_secret=access_token_secret)

    try:
        tweets_fetched = twitter_api.GetSearch(string_buzzword, count=100, lang = 'en')
        return [{"text": status.text, "label": None} for status in tweets_fetched]

    except:
        raise ValueError("Unfortunately, something went wrong.")

# create news content test set:

def news_test_set_content(string_buzzword, secret_api):
    url = 'https://newsapi.org/v2/everything?'

    newsapi = NewsApiClient(api_key=secret_api)

    # Get 100 articles from past 30 days:
    all_articles = newsapi.get_everything(q=string_buzzword,
                                          # sources='bbc-news,the-verge',
                                          # domains='bbc.co.uk,techcrunch.com',
                                          language='en',
                                          page_size = 100,
                                          sort_by='relevancy')

    return [all_articles['articles'][i]['description'] for i in range(len(all_articles['articles']))]

# create news url test set:
def news_test_set_url(string_buzzword, secret_api):
    url = 'https://newsapi.org/v2/everything?'

    newsapi = NewsApiClient(api_key=secret_api)

    # Get 100 articles from past 30 days:
    all_articles = newsapi.get_everything(q=string_buzzword,
                                          # sources='bbc-news,the-verge',
                                          # domains='bbc.co.uk,techcrunch.com',
                                          language='en',
                                          page_size = 100,
                                          sort_by='relevancy')

    return [all_articles['articles'][i]['url'] for i in range(len(all_articles['articles']))]

# process content using NLTK package by removing some punctuation, images, urls and hashtags from the tweets & news:
def process_tweet(tweet):
    _stopwords = set(stopwords.words('english') + list(punctuation) + ['AT_USER', 'URL'])

    tweet = tweet.lower()  # convert text to lower-case
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', tweet)  # remove URLs
    tweet = re.sub('@[^\s]+', 'AT_USER', tweet)  # remove usernames
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)  # remove the # in #hashtag
    tweet = word_tokenize(tweet)  # remove repeated characters (helloooooooo into hello)
    return [word for word in tweet if word not in _stopwords]

def Average(lst):
    return sum(lst) / len(lst)

# the following returns the average polarity for both news and twitter content for each buzzword:
def polarity_twitter_news(string_buzzword, consumer_key, consumer_secret, access_token_key, access_token_secret, secret_api):

    ### polarity twitter:

    test_set = twitter_test_set(string_buzzword, consumer_key, consumer_secret, access_token_key, access_token_secret)
    processed_test_set_twitter = [(process_tweet(tweet["text"]), tweet["label"]) for tweet in test_set]

    preprocessedTestSet_2 = []
    [preprocessedTestSet_2.append(processed_test_set_twitter[x][0]) for x in range(len(processed_test_set_twitter))]
    preprocessedTestSet_3 = []
    [preprocessedTestSet_3.append(TextBlob(' '.join(preprocessedTestSet_2[x]))) for x in
     range(len(preprocessedTestSet_2))]

    results_polarity_twitter = []
    [results_polarity_twitter.append(element.sentiment[0]) for element in preprocessedTestSet_3]

    ### polarity news:

    testDataSet_news = news_test_set_content(string_buzzword, secret_api)

    results_polarity_news = []

    [results_polarity_news.append(TextBlob(str(testDataSet_news[x])).sentiment[0]) for x in range(len(testDataSet_news))]

    return Average(results_polarity_twitter), Average(results_polarity_news)

# the following returns the average polarity for both news and twitter content for all the buzzwords:
def average_polarity(polarity_buzzwords):

    avg_pol_news = []
    avg_pol_twitter = []

    [avg_pol_twitter.append(polarity_buzzwords[x][0]) for x in range(len(polarity_buzzwords))]

    [avg_pol_news.append(polarity_buzzwords[x][1]) for x in range(len(polarity_buzzwords))]

    average_polarity_buzzwords = Average(avg_pol_news), Average(avg_pol_twitter)

    return list(average_polarity_buzzwords)

def visualisation_polarity(polarity_buzzwords):

    polarity_names = ["Polarity News", "Polarity Twitter"]

    fig = go.Figure(data=[
        go.Bar(name='News Sentiment', x=polarity_names,
               y=average_polarity(polarity_buzzwords))
    ])

    fig.update_layout(title_text='Polarity for Specific Keyword')

    fig.update_layout(template='simple_white',
                      title_text='Polarity Analysis Results',
                      yaxis=dict(
                          title=' -1 (Negative) : 1 (Positive) ',
                          titlefont_size=16,
                          tickfont_size=14, ),

                      )

    return fig.show()

# the following returns the average polarity for both news and tweets for each specific buzzword
def visualisation_polarity_each_buzzword(polarity_buzzwords, buzzwords):
    fig = make_subplots(
        rows=len(polarity_buzzwords), cols=1, )

    polarity_names = ["Polarity Twitter", "Polarity News"]
    for x in range(len(polarity_buzzwords)):
        buzzword_name = str(buzzwords[x])
        fig.add_trace(
            go.Bar(x=polarity_names, y=polarity_buzzwords[x], name=F" {buzzword_name} sector "),
            row=x + 1, col=1
        )

    fig.update_layout(title_text='Polarity for Specific Keyword')

    fig.update_layout(template='simple_white',
                      title_text='Polarity Break-Down per Buzzword',
                      yaxis=dict(
                          title=' -1 (Negative) : 1 (Positive) ',
                          titlefont_size=16,
                          tickfont_size=14, ),

                      )

    return fig

# the following returns news urls with lowest polarity score given a specific buzzword/sector:
def extract_news_low_polarity(number_news, string_buzzword, secret_api):

    testDataSet_news = news_test_set_content(string_buzzword, secret_api)
    testDataSet_url = news_test_set_url(string_buzzword, secret_api)

    lst_polarity = []

    [lst_polarity.append(TextBlob(str(testDataSet_news[x])).sentiment[0]) for x in range(len(testDataSet_news))]

    index_low_polarity_list = sorted(range(len(lst_polarity)), key=lambda sub: lst_polarity[sub])[:number_news]

    news_low_polarity = []
    [news_low_polarity.append(testDataSet_url[index_low_polarity_list[x]]) for x in range(len(index_low_polarity_list))]

    return print(news_low_polarity)

    #return print(testDataSet_news[index_low_polarity_list[0]]), print(testDataSet_url[index_low_polarity_list[0]], print(TextBlob(testDataSet_url[index_low_polarity_list[0]]).sentiment))

def news_flag_low_polarity(buzzwords, polarity, secret_api):
    news_url = []
    for x in range(len(buzzwords)):
        buzzword_name = str(buzzwords[x])
        if polarity[x][1] <= 0:
            print(F" News Polarity level for {buzzword_name} sector is low!")
            number_news = int(input(F'Enter number of news articles you want to read about {buzzword_name} sector: '))
            news_url.append(extract_news_low_polarity(number_news, buzzwords[x], secret_api))
        else:
            print(F" News Polarity level for {buzzword_name} sector is fine")

    # return print(news_url)

### Below is the code for the time series analysis. Due to restrictions of the News Api, we could only perform the analysis for the past 30 days on a daily basis:
### With the current restrictions we can't obtain data (news) older than 1 month.

# create news content test set with daily data:
def news_test_set_content_daily(date_to, date_from, buzzword, secret_api):
    url = 'https://newsapi.org/v2/everything?'

    newsapi = NewsApiClient(api_key=secret_api)

    all_articles = newsapi.get_everything(q=buzzword,
                                          # sources='bbc-news,the-verge',
                                          # domains='bbc.co.uk,techcrunch.com',
                                          from_param=date_from,
                                          to=date_to,
                                          language='en',
                                          sort_by='relevancy')

    return [all_articles['articles'][i]['description'] for i in range(len(all_articles['articles']))]


def get_dates_past_month(duration_days):
    lst_dates_past_month = []
    for x in range(duration_days):
        start_date = datetime.date.today()
        delta = dateutil.relativedelta.relativedelta(months=1)
        one_month_earlier = start_date - delta
        lst_dates_past_month.append(start_date - datetime.timedelta(x))
        # lst_dates_past_month.append(one_month_earlier + datetime.timedelta(x))

    return sorted(lst_dates_past_month, reverse=True)


def get_time_data_previous_month(duration_days):
    dates_past_month = get_dates_past_month(duration_days)

    lst_pairs = []
    for x in range(len(dates_past_month) - 1):
        pair = [dates_past_month[x], dates_past_month[x + 1]]
        lst_pairs.append(pair)

    return lst_pairs

# below returns the polarity score for a specific buzzword and a specific time range:
def polarity_score_daily(date_to, date_from, buzzword, secret_api):
    ### polarity news:

    testDataSet_news = news_test_set_content_daily(date_to, date_from, buzzword, secret_api)

    results_polarity_news = []

    [results_polarity_news.append(TextBlob(str(testDataSet_news[x])).sentiment[0]) for x in
     range(len(testDataSet_news))]

    return Average(results_polarity_news)

# below returns the daily polarity score for a specific buzzword and a specific time range:
def daily_news_polarity_scores_past_month(duration_days, buzzword, secret_api):
    lst_results_polarity_daily = []
    lst_pairs = get_time_data_previous_month(duration_days)
    for x in range(len(lst_pairs)):
        date_to = lst_pairs[x][0]
        date_from = lst_pairs[x][1]
        try:
            lst_results_polarity_daily.append(polarity_score_daily(date_to, date_from, buzzword, secret_api))
        except:
            lst_results_polarity_daily.append(0)

    return lst_results_polarity_daily


def visualisation_time_series_polarity_specific_buzzword(duration_days, buzzword, secret_api):
    fig = go.Figure(data=go.Scatter(x=get_dates_past_month(duration_days),
                                    y=daily_news_polarity_scores_past_month(duration_days, buzzword, secret_api)))

    fig.update_layout(title_text='News Polarity Evolution Past Month')

    fig.update_layout(template='simple_white',
                      title_text='Polarity Analysis Results',
                      yaxis=dict(
                          title=' -1 (Negative) : 1 (Positive) ',
                          titlefont_size=16,
                          tickfont_size=14, ), xaxis=dict(title='Last Month'), )

    return fig.show()

def polarity_time_series_visualisation(buzzwords, duration_days, secret_api):
    fig = go.Figure()
    for x in range(len(buzzwords)):
        buzzword_name = str(buzzwords[x])
        fig.add_trace(go.Scatter(
        x=get_dates_past_month(duration_days)[1:30],
        y=daily_news_polarity_scores_past_month(duration_days, buzzwords[x], secret_api)
        , mode='lines', name=F" {buzzword_name} sector "))
    fig.update_layout(template='simple_white', title_text="News Polarity Per Sector/Buzzword")
    return fig





