import torch
import numpy as np

def mean_absolute_percentage_error(y_pred, y_true, epsilon=1e-7):
    """Calculate Mean Absolute Percentage Error (MAPE)"""
    # Add epsilon to avoid division by zero
    y_true_safe = y_true + epsilon
    
    # Calculate absolute percentage error
    ape = torch.abs((y_true - y_pred) / y_true_safe)
    
    # Calculate mean
    mape = torch.mean(ape) * 100.0  # Convert to percentage
    
    return mape

def mean_squared_error(y_pred, y_true):
    """Calculate Mean Squared Error (MSE)"""
    return torch.mean((y_pred - y_true) ** 2)

def root_mean_squared_error(y_pred, y_true):
    """Calculate Root Mean Squared Error (RMSE)"""
    return torch.sqrt(mean_squared_error(y_pred, y_true))
