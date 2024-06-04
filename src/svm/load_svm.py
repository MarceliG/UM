import os

import joblib
from sklearn import svm


def load_svm_model(model_path: str) -> svm.SVC:
    """
    Loads an SVM model from a specified file.

    Args:
        model_path (str): The path to the file from which the model will be loaded.

    Returns:
        The loaded SVM model.
    """
    if not os.path.exists(model_path):
        raise_message = f"No model found at the specified path: {model_path}"
        raise FileNotFoundError(raise_message)

    model = joblib.load(model_path)

    if not isinstance(model, svm.SVC):
        raise TypeError("The loaded model is not an instance of svm.SVC")

    return model


model: svm.SVC = load_svm_model("/home/marceli/wsb/text_classification/data/models/default/linear.pkt")
print(model.kernel)
