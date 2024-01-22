import torch
from monai.networks.nets import SegResNet
from .constants import DEVICE


def init_model(weights_path):
    model = SegResNet(init_filters=32, in_channels=1, out_channels=105)
    model.load_state_dict(torch.load(weights_path, map_location=DEVICE))  # if no gpu use map_location="cpu"
    model.eval()
    return model
