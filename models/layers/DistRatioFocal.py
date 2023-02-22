import torch
from torch import nn
from models.layers.DistRatio import DistRatio

class DistRatioFocal:
    @staticmethod
    def apply(input1, input2, y, margin):
        o = DistRatio.apply(input1, input2, y, margin)
        return (1 - torch.exp(-o)) * o