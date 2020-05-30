import torch

from flask import Flask
from torch import nn

from hp import POS_WEIGHT

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

criterion = nn.CrossEntropyLoss(ignore_index=0)
binary_criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([POS_WEIGHT]).to(DEVICE))

app = Flask(__name__, static_url_path='')

from flask_app import routes
