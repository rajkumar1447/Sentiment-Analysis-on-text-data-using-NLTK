# Import necessary libraries.
from sentiment.data_loader import load_data
from sentiment.model import prepare_datasets, train_classifier, evaluate_classifier
from sentiment.analyzer import analyze_sentiment
import nltk

nltk.download('movie_reviews')
nltk.download('punkt')       # Sentence tokenizer
nltk.download('punkt_tab')   # 🔥 NEW dependency
nltk.download('stopwords')

def main():
    # Load dataset
    docs = load_data()

    # Prepare datasets
    train_set, test_set = prepare_datasets(docs)

    # Train classifier
    classifier = train_classifier(train_set)

    # Evaluate
    accuracy = evaluate_classifier(classifier, test_set)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Show informative features
    classifier.show_most_informative_features(10)

    # Test with custom input
    while True:
        text = input("\nEnter a sentence (or 'quit' to stop): ")
        if text.lower() == "quit":
            break
        sentiment = analyze_sentiment(text, classifier)
        print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
