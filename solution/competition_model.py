"""Define your architecture here."""
import torch
from models import SimpleNet
import torch.nn as nn
from torchvision import models
# TODO: VERIFY THIS LIBRARY
from collections import OrderedDict

save_path = 'checkpoints/competition_model.pt'

def fineTune_block(in_features):
    return nn.Sequential(OrderedDict([
        ('Linear1', nn.Linear(in_features, 624)),
        ('relu1', nn.SiLU()),
        ('Linear2', nn.Linear(624, 256)),
        ('relu2', nn.SiLU()),
        ('Linear3', nn.Linear(256, 64)),
        ('relu3', nn.SiLU()),
        ('Linear4', nn.Linear(64, 2)),
    ]))


def compModel(savenew=False, pretrained=False):
    # Based on: https://pytorch.org/vision/main/models.html
    model = models.efficientnet_v2_m(pretrained=pretrained) # weights are saved
    num_features = model.classifier[1].in_features
    tune_block = fineTune_block(in_features=num_features)
    model.classifier = tune_block # binary classification (num_of_class == 2)

    if savenew:
        torch.save({'model': model.state_dict()}, save_path)
    return model





def my_competition_model():
    """Override the model initialization here.

    Do not change the model load line.
    """
    # initialize your model:
    model = compModel()
    # load your model using exactly this line (don't change it):
    model.load_state_dict(torch.load('checkpoints/competition_model.pt')['model'])
    return model
