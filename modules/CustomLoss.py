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
    

class FocalMSELoss(nn.Module):
    # Basically MSE with a focal loss type added weight for emphasizing bad samples
    def __init__(self):
        super(FocalMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        # Get absolute differences 
        differences = torch.abs( y_pred - y_true )

        # Calculate weighting
        weight_factor = (1 - torch.exp(-differences))**2

        # Calculate MSE
        mse = differences**2

        # Calculate focal mse for all samples
        focal_mse_loss = weight_factor * mse

        return focal_mse_loss.mean()
