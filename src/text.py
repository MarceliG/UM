"""File contains all funtions to prepare or modify text."""

from typing import List, Tuple

from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split


def split_data(
    texts: csr_matrix,
    labels: List[int],
    train_size: float = 0.7,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[csr_matrix, csr_matrix, csr_matrix, List[int], List[int], List[int]]:
    """
    Splits data into training, validation, and test sets.

    Args:
        texts (csr_matrix): Feature matrix.
        labels (List[int]): Labels.
        train_size (float): Proportion of the data to include in the training set.
        val_size (float): Proportion of the data to include in the validation set.
        test_size (float): Proportion of the data to include in the test set.
        random_state (int): Random seed for reproducibility.

    Returns:
        Split data into training, validation, and test sets.

    Raises:
        ValueError: If the sum of train_size, val_size, and test_size is not equal to 1.0.
    """
    if train_size + val_size + test_size != 1.0:
        raise ValueError("Sum of train_size, val_size, and test_size must be 1.0")

    texts_train, texts_temp, labels_train, labels_temp = train_test_split(
        texts, labels, train_size=train_size, random_state=random_state
    )
    relative_test_size = test_size / (test_size + val_size)
    texts_val, texts_test, labels_val, labels_test = train_test_split(
        texts_temp,
        labels_temp,
        test_size=relative_test_size,
        random_state=random_state,
    )

    return (
        texts_train,
        texts_val,
        texts_test,
        labels_train,
        labels_val,
        labels_test,
    )
