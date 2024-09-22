import numpy as np
import pandas as pd

import torch
import torch.nn as nn

class WeightedMSELoss(nn.Module):
    def __init__(self, weight_threshold, weight_multiplier):
        super(WeightedMSELoss, self).__init__()
        self.weight_threshold = weight_threshold
        self.weight_multiplier = weight_multiplier

    def forward(self, y_pred, y_true):
        # Get absolute difference of predictions and targets
        differences = torch.abs( y_pred - y_true )

        # Assign increased weight to small deviations to push learning
        weights = torch.where(differences < self.weight_threshold, self.weight_multiplier, 1.0)

        # Calculate the squared errors, then take the weighted mean
        squared_errs = (y_pred - y_true) ** 2
        weighted_sum = weights * squared_errs

        return torch.mean(weighted_sum)