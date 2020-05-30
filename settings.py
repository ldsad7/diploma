import logging

import nltk
import torch

nltk.download('punkt')

from functools import lru_cache
from torch import nn
from flask_app import DEVICE
from model import BertMultiTaskLearning

logging.info("Detected ", torch.cuda.device_count(), "GPUs!")


@lru_cache(maxsize=-1, typed=False)
def load_model(model_type, **dct):
    """ load the the model """

    model = BertMultiTaskLearning.from_pretrained('bert', **dct)
    model = nn.DataParallel(model)
    model.to(DEVICE)
    model_path = f"trained_models/{model_type}.pt"
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    return model
