import argparse

from svm import run_svm


def run_model(model_type: str, param_type: str):
    if model_type == "svm":
        if param_type == "default":
            print("Running SVM with default parameters...")
        elif param_type == "best":
            print("Running SVM with best parameters...")
        run_svm(model_type=param_type)
    elif model_type == "bert":
        if param_type == "default":
            print("Running BERT with default parameters...")
        elif param_type == "best":
            print("Running BERT with best parameters...")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Text Classification")
    parser.add_argument(
        "-dd",
        "--download-data",
        action="store_true",
        help="If passed, the dataset for text classification will be downloaded.",
    )
    parser.add_argument(
        "-pp",
        "--pre-processing",
        action="store_true",
        help="If passed, run text processing and save the processed text.",
    )
    parser.add_argument(
        "--svm",
        choices=["default", "best"],
        nargs="*",
        default=["default", "best"],
        help="Train SVM model. Options: 'default', 'best'. You can specify both.",
    )
    parser.add_argument(
        "--bert",
        choices=["default", "best"],
        nargs="*",
        help="Train BERT model. Options: 'default', 'best'. You can specify both.",
    )

    args = parser.parse_args()
    if args.download_data:
        print("Downloading data...")

    if args.pre_processing:
        print("Running text processing and saving the processed text...")

    for model_type in ["svm", "bert"]:
        params = getattr(args, model_type)
        if params:
            for param in params:
                run_model(model_type, param)

    # always display classification report


if __name__ == "__main__":
    main()
