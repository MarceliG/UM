import os
from typing import Dict, Optional, Union

import nltk
import numpy as np
import optuna
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score

from configuration import DATASETS_PREPROCESED_PATH
from data_manager import load_dataset_from_disc
from preprocesing import split_data
from svm import save_svm_model


class SVMclassifier:
    """Class to find the best parameters of the SVM model."""

    def __init__(self) -> None:
        pass

    def create_model(
        self,
        c: float = 1.0,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        coef0: float = 0.0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 0.001,
        cache_size: int = 200,
        class_weight: Union[Dict[int, float], str, None] = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: Optional[int] = None,
    ) -> Dict:
        """
        Creates an SVM model with specified hyperparameters.

        Args:
            c: Regularization parameter. The strength of the regularization is inversely proportional to C.
            kernel: Specifies the kernel type to be used in the algorithm.
                It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed', or a callable.
            degree: Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.
            gamma: Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
                If 'scale', it uses 1 / (n_features * X.var()), if 'auto', it uses 1 / n_features.
            coef0: Independent term in kernel function. It is only significant in 'poly' and 'sigmoid'.
            shrinking: Whether to use the shrinking heuristic.
            probability: Whether to enable probability estimates. This must be enabled prior to calling fit,
                and will slow down that method.
            tol: Tolerance for stopping criterion.
            cache_size: Size of the kernel cache (in MB).
            class_weight: Set the parameter C of class i to class_weight[i]*C for SVC. If not given,
                all classes are supposed to have weight one. The 'balanced' mode uses the values of y
                to automatically adjust weights inversely proportional to class frequencies.
            verbose: Enable verbose output. Note that this setting takes advantage of a per-process runtime setting
                in libsvm that, if enabled, may not work properly in a multithreaded context.
            max_iter: Hard limit on iterations within solver, or -1 for no limit.
            decision_function_shape: Whether to return a one-vs-rest ('ovr') decision function of shape
                (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo')
                decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).
            break_ties: If true, `decision_function_shape`='ovr', and number of classes > 2,
                `predict` will break ties according to the confidence values.
            random_state: Controls the pseudo random number generation for shuffling the data for probability estimates.

        Returns:
            An SVM model configured with the specified parameters.
        """
        model = svm.SVC(
            C=c,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )

        return {
            "model": model,
            "parameters": {
                "C": c,
                "kernel": kernel,
                "degree": degree,
                "gamma": gamma,
                "coef0": coef0,
                "shrinking": shrinking,
                "probability": probability,
                "tol": tol,
                "cache_size": cache_size,
                "class_weight": class_weight,
                "verbose": verbose,
                "max_iter": max_iter,
                "decision_function_shape": decision_function_shape,
                "break_ties": break_ties,
                "random_state": random_state,
            },
        }

    def create_models_with_all_kernels(self) -> Dict[str, svm.SVC]:
        """
        Creates SVM models with each kernel type.

        Returns:
            A dictionary of SVM models with keys as kernel names and values as the corresponding SVM model.
        """
        kernels = ["linear", "poly", "rbf", "sigmoid"]
        models = {}

        for kernel in kernels:
            models[kernel] = self.create_model(kernel=kernel)

        return models

    def find_best_model(self, texts: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Use GridSearchCV to optimize hyperparameters.

        Args:
            texts: Feature data.
            labels: Target labels.

        Returns:
            A dictionary containing the best model with optimized parameters.
        """
        param_grid = {
            "C": [0.001, 0.01, 0.1, 1, 10, 100],
            "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
            "kernel": ["linear", "poly", "rbf", "sigmoid"],
            "degree": [1, 2, 3, 4, 5],
            "coef0": [0.0, 0.1, 0.5, 1.0],
        }

        grid = GridSearchCV(svm.SVC(), param_grid, verbose=3, cv=5)
        grid.fit(texts, labels)

        best_params = grid.best_params_
        best_model = grid.best_estimator_

        return {"best_model": best_model, "best_parameters": best_params}


def run_svm(model_type: str, percentage_dataset: float = 100):
    """Run SVM function."""

    nltk.download("stopwords")

    dataset = load_dataset_from_disc(os.path.join(DATASETS_PREPROCESED_PATH, "raw_review_All_Beauty"))
    df = dataset.to_pandas()

    texts = df["text"]
    labels = df["rating"]

    # Get % subset of dataset
    subset_size = round(len(df) * percentage_dataset / 100)
    print(f"Size of dataset: {subset_size}")
    subset_df = df.sample(n=subset_size, random_state=42)

    texts = subset_df["text"]
    labels = subset_df["rating"]

    stop_words = list(stopwords.words("english"))

    vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=1000, max_df=0.90, min_df=2)

    x_tfidf = vectorizer.fit_transform(texts)

    (
        texts_train,
        texts_val,
        texts_test,
        labels_train,
        labels_val,
        labels_test,
    ) = split_data(texts=x_tfidf, labels=labels)

    svm_classifier = SVMclassifier()

    if model_type == "default":
        models = svm_classifier.create_models_with_all_kernels()
        for model_name, model in models.items():
            print(f"Training: {model_name}")
            model.get("model").fit(texts_train, labels_train)
            labels_pred = model.get("model").predict(texts_test)
            print(f"*****{model_name}*****")
            print("Classification Report:")
            print(
                classification_report(
                    labels_test,
                    labels_pred,
                    zero_division=0,
                )
            )
            save_svm_model(
                model=model.get("model"),
                model_name=model_name,
                model_type="default",
            )
    elif model_type == "best":
        model = svm_classifier.find_best_model(texts=texts_train, labels=labels_train)
        labels_pred_best_model = model.get("best_model").predict(texts_test)
        print()
        print("*****best_model*****")
        print(model.get("best_model").kernel)
        print(model.get("best_parameters"))
        print("Classification Report:")
        print(
            classification_report(
                labels_test,
                labels_pred_best_model,
                zero_division=0,
            )
        )
        save_svm_model(
            model=model.get("model"),
            model_name=model.get("best_model").kernel,
            model_type="best",
        )
    elif model_type == "custom":
        custom_model = svm_classifier.create_model(c=100, coef0=0.5, degree=5, gamma=0.01, kernel="poly")
        custom_model.get("model").fit(texts_train, labels_train)
        labels_pred_custom_model = custom_model.get("model").predict(texts_test)
        print()
        print("*****custom_model*****")
        print(custom_model)
        print("Classification Report:")
        print(
            classification_report(
                labels_test,
                labels_pred_custom_model,
                zero_division=0,
            )
        )
        save_svm_model(
            model=custom_model,
            model_name=custom_model.kernel,
            model_type="custom",
        )
