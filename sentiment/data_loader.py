# Import necessary libraries.
import nltk
from nltk.corpus import movie_reviews
import random

# Download required datasets (only if not already installed)
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')

# Function to load the dataset.
def load_data():
    docs = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    random.shuffle(docs)
    return docs
