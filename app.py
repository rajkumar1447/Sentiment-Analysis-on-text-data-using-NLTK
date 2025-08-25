# Import necesaary libraries.
import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from nltk.corpus import stopwords
import random

# Download the NLTK data files.
nltk.download('movie_reviews')
nltk.download('punk')
nltk.download('stopwords')

# Preprocess the dataset and extract features.
def extract_features(words):
    return {word: True for word in words}

# Load and pre process the data set from NLTK
docs = [(list(movie_reviews.words(fileid)), category)
        for category in movie_reviews.categories()
        for fileid in movie_reviews.fileids(category)]

# Shuffle the dataset to ensure random distribution
random.shuffle(docs)

# Prepare the data set for training and testing.
features = [(extract_features(d), c) for (d, c) in docs]
train_set, test_set = features[:1600], features[1600:]

# Train the NAvieBayesClassifier.
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the Classifier.
accuracy = nltk_accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most Imfomative features.
classifier.show_most_informative_features(10)

# Test on new Input Sentances.
def analyze_sentiment(text):
    # Tokenize and remove stopwords
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stopwords.words('english')]

    # Predict sentiment
    features = extract_features(words)
    return classifier.classify(features)



