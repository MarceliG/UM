import os

HOME_PATH = os.getcwd()
DATA_PATH = os.path.join(HOME_PATH, "data")
MODELS_PATH = os.path.join(DATA_PATH, "models")
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
DATASETS_PREPROCESED_PATH = os.path.join(DATASETS_PATH, "preprocesed")
IMAGES_PATH = os.path.join(DATASETS_PATH, "images")

HOME_PATH = os.getcwd()
folders_to_create = [
    DATA_PATH := os.path.join(HOME_PATH, "data"),
    MODELS_PATH := os.path.join(DATA_PATH, "models"),
    DATASETS_PATH := os.path.join(DATA_PATH, "datasets"),
    DATASETS_PREPROCESED_PATH := os.path.join(DATASETS_PATH, "preprocesed"),
    IMAGES_PATH := os.path.join(DATASETS_PATH, "images"),
]

for folder in folders_to_create:
    os.makedirs(folder, exist_ok=True)
