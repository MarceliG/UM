import os

import torch
from transformers import BertModel

from configuration import MODELS_PATH

def save_bert_model(model: BertModel, model_name: str):
    """
    Saves as BERT model to a specified file.
    
    Args:
        model: The BERT model to be saved
        model_name: The name of the model will be saved.
    """
    models_path = os.path.join(MODELS_PATH, 'bert')
    os.makedirs(models_path, exist_ok=True)
    if not model_name.endswith('.pth'):
        model_name += ".pth"
    path_to_save = os.path.join(models_path, model_name)
    torch.save(model.state_dict(), path_to_save)