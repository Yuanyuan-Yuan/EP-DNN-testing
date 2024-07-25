import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from models.face import FaceNet

def get_model(model_name):
    if model_name == 'FaceNet':
        PATH = './trained_models/face.pt'
        model = FaceNet(n_class=10178)
        model.exp_name = 'face'
        model.load_state_dict(torch.load(PATH)['classifier'])
    elif model_name == 'alexnet':
        PATH = './trained_models/animal-alexnet.pt'
        model = torchvision.models.alexnet(pretrained=False)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
        model.exp_name = 'animal-alexnet'
        model.load_state_dict(torch.load(PATH)['model'])
    elif model_name == 'efficientnet_b0':
        PATH = './trained_models/car-efficientnet_b0.pt'
        model = torchvision.models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 196)
        model.exp_name = 'car-efficientnet_b0'
        model.load_state_dict(torch.load(PATH)['model'])
    else:
        model = torchvision.models.__dict__[model_name](pretrained=True)
        model.exp_name = model_name
    model.eval()
    return model

def get_repaired_model(model_name):
    if model_name == 'FaceNet':
        PATH = './repaired_models/face.pt'
        model = FaceNet(n_class=8192)
        model.exp_name = 'face'
        model.load_state_dict(torch.load(PATH)['model'])
    elif model_name == 'alexnet':
        PATH = './repaired_models/animal-alexnet.pt'
        model = torchvision.models.alexnet(pretrained=False)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 3)
        model.exp_name = 'animal-alexnet'
        model.load_state_dict(torch.load(PATH)['model'])
    elif model_name == 'efficientnet_b0':
        PATH = './repaired_models/car-efficientnet_b0.pt'
        model = models.efficientnet_b0(pretrained=False)
        num_ftrs = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(num_ftrs, 196)
        model.exp_name = 'car-efficientnet_b0'
        model.load_state_dict(torch.load(PATH)['model'])
    else:
        model = torchvision.models.__dict__[model_name](pretrained=False)
        model.exp_name = model_name
        PATH = './repaired_models/%s.pt' % model_name
        model.load_state_dict(torch.load(PATH)['model'])
    model.eval()
    return model
