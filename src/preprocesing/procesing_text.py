import os
import re

import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from data_manager import load_dataset_from_disc, save_dataset

DATA_PATH = os.path.join(os.getcwd(), "data")
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
DATASETS_PREPROCESED_PATH = os.path.join(DATASETS_PATH, "preprocesed")
DATASETS_TOKENIZED_PATH = os.path.join(DATASETS_PATH, "tokenized")
IMAGES_PATH = os.path.join(DATASETS_PATH, "images")


def plot_text_length_distribution(dataset, dataset_name):
    os.makedirs(IMAGES_PATH, exist_ok=True)
    # Pobierz kolumnę tekstów
    texts = dataset["text"]

    # Oblicz długość każdego tekstu (liczbę słów lub znaków)
    text_lengths = [len(text.split()) for text in texts]  # Liczba słów

    # Wygeneruj histogram długości tekstów
    plt.hist(text_lengths, bins="auto", edgecolor="black", alpha=0.7)
    plt.xlabel("Text Length")
    plt.ylabel("Frequency")
    plt.title("Text Length Distribution")
    path_to_save = os.path.join(IMAGES_PATH, dataset_name + ".jpeg")
    plt.savefig(path_to_save)  # Zapisz wykres jako obraz


def map_rating(rating):
    return 0 if rating <= 2.5 else 1


def balance_dataset(dataset):
    dataset_pandas = dataset.to_pandas()

    dataset_pandas["text_length"] = dataset_pandas["text"].apply(lambda x: len(x.split()))
    dataset_pandas = dataset_pandas.loc[dataset_pandas["text_length"] > 0]
    # Oblicz IQR
    q1 = np.percentile(dataset_pandas["text_length"], 25)
    q3 = np.percentile(dataset_pandas["text_length"], 75)
    iqr = q3 - q1

    # Granice IQR
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    print(f"Deleting data that is shorter than: {lower_bound}")
    print(f"Deleting data that is longer than: {upper_bound}")
    # Usuń wiersze, które mają długość tekstu poza granicami IQR
    filtered_df = dataset_pandas[
        (dataset_pandas["text_length"] >= lower_bound) & (dataset_pandas["text_length"] <= upper_bound)
    ]
    filtered_df = filtered_df.drop(columns=["text_length"])
    filtered_df = filtered_df.reset_index(drop=True)
    return Dataset.from_pandas(filtered_df)


def preprocesing_dataset(dataset_name):
    print("Running text processing")
    dataset_raw = load_dataset_from_disc(os.path.join(DATASETS_PATH, dataset_name)).select_columns(["rating", "text"])

    plot_text_length_distribution(dataset_raw, dataset_name)

    balanced_dataset = balance_dataset(dataset_raw)

    plot_text_length_distribution(balanced_dataset, dataset_name + "_balanced")

    # change rating to binary
    dataset_filtered = balanced_dataset.map(lambda example: {"rating": map_rating(example["rating"])})

    print("Saving dataset...")
    save_dataset(
        dataset_filtered,
        dataset_name,
        path=DATASETS_PREPROCESED_PATH,
    )
    print("Saved")
    tokenize_svm_text(dataset_filtered, dataset_name)


def batch_generator(texts, batch_size=100):
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


def tokenize_svm_text(dataset, dataset_name, batch_size=1000):
    print("Tokenize...")

    texts = dataset["text"]
    labels = dataset["rating"]

    nltk.download("stopwords")
    stop_words = list(stopwords.words("english"))
    vectorizer = TfidfVectorizer(stop_words=stop_words, )

    total_texts = len(texts)
    progress_bar = tqdm(total=total_texts, desc="Tokenizing")

    # Tokenizujemy i przetwarzamy teksty w partiach
    tokenized_texts = []
    for batch_texts in batch_generator(texts, batch_size):
        x_tfidf = vectorizer.fit_transform(batch_texts)
        features = vectorizer.get_feature_names_out()
        batch_tokenized_texts = [dict(zip(features, row)) for row in x_tfidf.toarray()]
        tokenized_texts.extend(batch_tokenized_texts)

        # Aktualizacja paska postępu
        progress_bar.update(len(batch_texts))

    # Zamknięcie paska postępu
    progress_bar.close()

    # Konwertujemy rzadką macierz do postaci listy słowników
    features = vectorizer.get_feature_names_out()
    tokenized_texts = [dict(zip(features, row)) for row in x_tfidf.toarray()]

    # Tworzymy nowy zestaw danych z przetworzonymi danymi
    processed_dataset = Dataset.from_dict({"tokenized_text": tokenized_texts, "rating": labels})

    print("Saving tokenized text")
    save_dataset(processed_dataset, dataset_name, DATASETS_TOKENIZED_PATH)
    print("Saved")
