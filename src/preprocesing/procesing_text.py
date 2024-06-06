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


def plot_text_length_distribution(original_text_lengths, balanced_text_lengths, dataset_name):
    os.makedirs(IMAGES_PATH, exist_ok=True)

    # Stwórz dwa subploty dla porównania
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Pierwszy subplot dla oryginalnych danych
    ax1.hist(original_text_lengths, bins=250, edgecolor="black")
    ax1.set_xlabel("Text Length")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Original Text Length Distribution")
    ax1.set_xlim(0, 300)

    # Drugi subplot dla danych po równoważeniu
    ax2.hist(balanced_text_lengths, bins=10, edgecolor="black")
    ax2.set_xlabel("Text Length")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Balanced Text Length Distribution")
    ax2.set_xlim(0, 300)

    # Zapisz wykresy jako obraz
    path_to_save = os.path.join(IMAGES_PATH, dataset_name + "_comparison.jpeg")
    plt.savefig(path_to_save)
    plt.close()


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
    print(dataset_raw)

    original_text_lengths = [len(text.split()) for text in dataset_raw["text"]]

    balanced_dataset = balance_dataset(dataset_raw)
    print(balanced_dataset)

    balanced_text_lengths = [len(text.split()) for text in balanced_dataset["text"]]

    # Porównaj rozkład długości tekstu przed i po równoważeniu danych
    plot_text_length_distribution(original_text_lengths, balanced_text_lengths, dataset_name)

    # change rating to binary
    dataset_filtered = balanced_dataset.map(lambda example: {"rating": map_rating(example["rating"])})

    print("Saving dataset...")
    save_dataset(
        dataset_filtered,
        dataset_name,
        path=DATASETS_PREPROCESED_PATH,
    )
    print("Saved")
