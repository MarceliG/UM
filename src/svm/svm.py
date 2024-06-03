from typing import Dict, Optional, Union

import nltk
import numpy as np
import optuna
import pandas as pd
from nltk.corpus import stopwords
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

from preprocesing import split_data


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

    def find_best_model(self, texts: np.ndarray, labels: np.ndarray, n_trials: int = 100) -> Dict:
        """
        Use Optuna to optimize hyperparameters.

        Args:
            texts: Feature data.
            labels: Target labels.
            n_trials: Number of trials for the optimization.

        Returns:
            A dictionary containing the best model with optimized parameters.
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: self.objective(trial, texts, labels), n_trials=n_trials)

        best_params = study.best_params
        best_model = self.create_model(
            c=best_params["C"],
            kernel=best_params["kernel"],
            degree=best_params.get("degree", 3),
            gamma=best_params.get("gamma", "scale"),
            coef0=best_params.get("coef0", 0.0),
            shrinking=best_params.get("shrinking", True),
            probability=best_params.get("probability", False),
            tol=best_params.get("tol", 0.001),
            cache_size=best_params.get("cache_size", 200),
            class_weight=best_params.get("class_weight"),
            verbose=best_params.get("verbose", False),
            max_iter=best_params.get("max_iter", -1),
            decision_function_shape=best_params.get("decision_function_shape", "ovr"),
            break_ties=best_params.get("break_ties", False),
            random_state=best_params.get("random_state"),
        )
        return {"best_model": best_model["model"], "best_parameters": best_model["parameters"]}

    def objective(self, trial: optuna.Trial, texts: np.ndarray, labels: np.ndarray) -> float:
        """
        Objective function for optimization.

        Args:
            trial: Optuna's Trial object.
            texts: Feature data.
            labels: Target labels.

        Returns:
            The accuracy score of the model with current hyperparameters.
        """
        c = trial.suggest_loguniform("C", 1e-10, 1e10)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        degree = trial.suggest_int("degree", 1, 10)
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        coef0 = trial.suggest_uniform("coef0", -10, 10)

        model = self.create_model(
            c=c,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=True,
            probability=False,
            tol=0.001,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=-1,
            decision_function_shape="ovr",
            break_ties=False,
            random_state=None,
        )
        scores = cross_val_score(model["model"], texts, labels, cv=5)
        return scores.mean()


def run_svm(model_type: str):
    """Run SVM function."""

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

    svm_classifier = SVMclassifier()

    if model_type == "default":
        models = svm_classifier.create_models_with_all_kernels()
        for model_name, model in models.items():
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
    elif model_type == "best":
        best_model_dict = svm_classifier.find_best_model(texts=texts_train, labels=labels_train)
        best_model_dict.get("best_model").fit(texts_train, labels_train)
        labels_pred_best_model = best_model_dict.get("best_model").predict(texts_test)
        print("*****best_model*****")
        print("Classification Report:")
        print(
            classification_report(
                labels_test,
                labels_pred_best_model,
                zero_division=0,
            )
        )
        print(best_model_dict)



# Upewnienie się, że dane są zbalansowane
# zapisz modele default i model best oraz parametry jakie mają

