import csv
import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go
from colorama import Fore,Style
from typing import Dict
import streamlit as st
nltk.download('vader_lexicon')

def extract_video_id(youtube_link):
    video_id_regex = r"^(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(video_id_regex,youtube_link)
    if match:
        video_id = match.group(1)
        return video_id
    else:
        return None
    
def analyze_sentiment(csv_file):
    sid= SentimentIntensityAnalyzer()

    comments =[]
    with open(csv_file,'r',encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            comments.append(row['Comment'])

    num_neutral =0
    num_positive =0
    num_negative =0
    for comment in comments:
        sentiment_scores = sid.polarity_scores(comment)
        if sentiment_scores['compound'] == 0.0:
            num_neutral += 1

        elif sentiment_scores['compound'] > 0.0:
            num_positive += 1
        else:
            num_negative += 1

    results = {'num_neutral':num_neutral,'num_negative':num_negative,'num_positive':num_positive}  
    return results

def bar_chart(csv_file: str) ->None:
    results = analyze_sentiment(csv_file)
    df = pd.DataFrame({
        'Sentiment': ['Neutral', 'Negative', 'Positive'],
        'Number of Comments': [results['num_neutral'], results['num_negative'], results['num_positive']]
    })  

    fig = px.bar(df, x='Sentiment', y='Number of Comments', color='Sentiment',title='Sentiment Analysis Results')
    fig.update_layout(title_font=dict(size=20))
    st.plotly_chart(fig,use_container_width = True)


def plot_sentiment(csv_file: str) -> None:
    results = analyze_sentiment(csv_file)

    labels = ['Neutral', 'Positive', 'Negative']
    values = [results['num_neutral'], results['num_positive'], results['num_negative']]
    colors = ['yellow', 'green', 'red']

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, textinfo='label+percent', marker=dict(colors=colors))])
    fig.update_layout(title={'text': 'Sentiment Analysis Results', 'font': {'size': 20, 'family': 'Arial', 'color': 'grey'}, 'x': 0.5})
    st.plotly_chart(fig)


def create_scatterplot(csv_file: str, x_column: str, y_column: str) -> None:
    data = pd.read_csv(csv_file)
    fig = px.scatter(data, x=x_column, y=y_column, color='Category')

    fig.update_layout(title='Scatter Plot', xaxis_title=x_column, yaxis_title=y_column, font=dict(size=18))
    st.plotly_chart(fig, use_container_width=True)


def print_sentiment(csv_file: str) -> None:
    results = analyze_sentiment(csv_file)
    num_positive = results['num_positive']
    num_negative = results['num_negative']

    if num_positive > num_negative:
        overall_sentiment = 'POSITIVE'
        color = Fore.GREEN
    elif num_negative > num_positive:
        overall_sentiment = 'NEGATIVE'
        color = Fore.RED
    else:
        overall_sentiment = 'NEUTRAL'
        color = Fore.YELLOW

    print('\n' + Style.BRIGHT + color + overall_sentiment.upper().center(50, ' ') + Style.RESET_ALL)

