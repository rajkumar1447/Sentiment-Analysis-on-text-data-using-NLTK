# Import necessary packages.
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy as nltk_accuracy
from .features import extract_features

# Function to prepare the dataset.
def prepare_datasets(docs, split_ratio=0.8):
    features = [(extract_features(d), c) for (d, c) in docs]
    split_index = int(len(features) * split_ratio)
    train_set, test_set = features[:split_index], features[split_index:]
    return train_set, test_set

# Function to train the model.
def train_classifier(train_set):
    classifier = NaiveBayesClassifier.train(train_set)
    return classifier

# Function to evaluate the model.
def evaluate_classifier(classifier, test_set):
    accuracy = nltk_accuracy(classifier, test_set)
    return accuracy
