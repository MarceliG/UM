import os

import joblib
from sklearn import svm

from configuration import MODELS_PATH


def save_svm_model(model: svm.SVC, model_name: str, model_type: str):
    """
    Saves an SVM model to a specified file.

    Args:
        model: The SVM model to be saved.
        model_name: The name of the model will be saved.
        model_type: The SVM model type `default` or `best`.
    """
    models_path = os.path.join(MODELS_PATH, model_type)
    os.makedirs(models_path, exist_ok=True)
    if not model_name.endswith(".pkt"):
        model_name += ".pkt"
    path_to_save = os.path.join(models_path, model_name)
    joblib.dump(model, path_to_save)
