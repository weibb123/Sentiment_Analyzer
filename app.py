import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from flask import Flask, render_template, url_for, request
import joblib
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)



analyzer = SentimentIntensityAnalyzer()


# regex for mentions and links in tweets
def preprocess(comment, stem=False):
     # getting rid of special characters might occur in text
     # will use this function on the text before passing in the algorithm
    regex = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
    english_stopwords = stopwords.words('english')
    comment = re.sub(regex, ' ', str(comment).lower()).strip()
    tokens = []
    for token in comment.split():
        if token not in english_stopwords:
            tokens.append(token)
    return " ".join(tokens)

def sentiment_score(x):
    if x >= 0.05:
        return "positive"
    elif x <= -0.05:
        return "negative"
    else:
        return "neutral"

def get_score(inputs):
    score = analyzer.polarity_scores(inputs)
    output = score['compound']
    sentiment = sentiment_score(output)
    return sentiment

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
# PUT requests to receive sentences from a user.
def predict():
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        myprediction = get_score(data)
    return render_template('index.html', prediction = f'Your input: "{message}" is {myprediction}')



if __name__ == '__main__':
    app.run(debug=True)

