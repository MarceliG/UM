import nltk
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC

from text import split_data


def main():
    """Main SVM function."""

    nltk.download("stopwords")

    # Replace with the proper texts
    data = {
        "text": [
            "I love this movie, it was fantastic!",
            "I hate this movie, it was terrible!",
            "This film was amazing, I enjoyed it a lot.",
            "What a bad movie, I did not like it.",
            "Great plot and excellent acting!",
            "Worst film ever, completely awful.",
            "It was an okay movie, nothing special.",
            "The storyline was very boring and dull.",
            "Loved the movie, it was wonderful!",
            "Terrible film, I disliked it a lot.",
            "Fantastic movie with great acting!",
            "Awful movie, not worth watching.",
            "One of the best movies I've seen.",
            "Really bad film, don't recommend it.",
            "Enjoyed every moment of the movie!",
            "The movie was very disappointing.",
        ],
        "label": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    }

    df = pd.DataFrame(data)
    texts = df["text"]
    labels = df["label"]

    stop_words = list(stopwords.words("english"))
    vectorizer = TfidfVectorizer(stop_words=stop_words)

    x_tfidf = vectorizer.fit_transform(texts)

    (
        texts_train,
        texts_val,
        texts_test,
        labels_train,
        labels_val,
        labels_test,
    ) = split_data(texts=x_tfidf, labels=labels)

    model = SVC(kernel="linear")
    model.fit(texts_train, labels_train)

    labels_pred = model.predict(texts_test)

    print("Accuracy:", accuracy_score(labels_test, labels_pred))
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))


if __name__ == "__main__":
    main()
