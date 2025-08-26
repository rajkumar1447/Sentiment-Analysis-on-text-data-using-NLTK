# Import necessary libraries.
import nltk
from nltk.corpus import stopwords
from .features import extract_features

# Function to analyse the sentiment.
def analyze_sentiment(text, classifier):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]
    features = extract_features(words)
    return classifier.classify(features)
