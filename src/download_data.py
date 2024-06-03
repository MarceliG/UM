import os
from typing import Optional

from datasets import Dataset, load_dataset


def download_dataset(category: str = "raw_review_All_Beauty") -> Dataset:
    """
    Downloads a dataset from the McAuley-Lab/Amazon-Reviews-2023 repository.

    Args:
        category: The category of the dataset to download. Defaults to "raw_review_All_Beauty".

    Returns:
        datasets.Dataset: The downloaded dataset.
    """
    print(f"Downloading datasets: McAuley-Lab/Amazon-Reviews-2023/{category}")
    return load_dataset("McAuley-Lab/Amazon-Reviews-2023", category, split="full", trust_remote_code=True)


def save_dataset(dataset: Dataset, path: Optional[str] = None):
    """
    Saves a dataset to the specified path or the current working directory.

    Args:
        dataset (datasets.Dataset): The dataset to save.
        path (str, optional): The path where the dataset will be saved. If None, saves to the current working directory.
            Defaults to None.
    """
    if path is None:
        path = os.path.join(os.getcwd(), "data", "datasets")
    print(f"Saving dataset to: {path}")
    dataset.save_to_disk(path)


def download():
    """Downloads a dataset and saves it to disk."""
    dataset = download_dataset()
    save_dataset(dataset=dataset)
