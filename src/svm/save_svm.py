import os

import joblib

home = os.getcwd()
data = os.path.join(home, "data")


def save_svm_model(models):
    # This function will be saving svm model
    print(data)
    # joblib.dump(model, "model.pkl")
