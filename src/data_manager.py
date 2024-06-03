import os
from typing import Optional

from datasets import Dataset, load_dataset, load_from_disk

DATASET_PATH = os.path.join(os.getcwd(), "data", "datasets")


def download_dataset(category: str) -> Dataset:
    """
    Downloads a dataset from the McAuley-Lab/Amazon-Reviews-2023 repository.

    Args:
        category: The category of the dataset to download.

    Returns:
        datasets.Dataset: The downloaded dataset.
    """
    print(f"Downloading datasets: McAuley-Lab/Amazon-Reviews-2023/{category}")
    return load_dataset("McAuley-Lab/Amazon-Reviews-2023", category, split="full", trust_remote_code=True)


def save_dataset(dataset: Dataset, dataset_name: str, path: Optional[str] = None) -> None:
    """
    Saves a dataset to the specified path or the current working directory.

    Args:
        dataset (datasets.Dataset): The dataset to save.
        dataset_name (str): Dataset name to save.
        path (str, optional): The path where the dataset will be saved. If None, saves to the current working directory.
            Defaults to None.
    """
    if path is None:
        path = os.path.join(DATASET_PATH, dataset_name)
    print(f"Saving dataset to: {path}")
    dataset.save_to_disk(path)


def load_dataset_from_disc(dataset_name: str) -> Dataset:
    """
    Loads a dataset from disk.

    Args:
        dataset_name (str): The name of the dataset to load.

    Returns:
        datasets.Dataset: The loaded dataset.
    """
    path = os.path.join(DATASET_PATH, dataset_name)
    return load_from_disk(path)


def download():
    """Downloads a dataset and saves it to disk."""
    category = "raw_review_All_Beauty"
    dataset = download_dataset(category)
    save_dataset(dataset=dataset, dataset_name=category)
