import numpy as np
import torch


def calculate_normalisation_params(train_loader, test_loader):
    """
    Calculate the mean and standard deviation of each channel
    for all observations in training and test datasets. The
    results can then be used for normalisation
    """ 
    chan0 = np.array([])
    chan1 = np.array([])
    chan2 = np.array([])
    
    for i, data in enumerate(train_loader, 0):
        images, _ = data
        chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
        chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
        chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
        
    for i, data in enumerate(test_loader, 0):
        images, _ = data
        chan0 = np.concatenate((chan0, images[:, 0, :, :].cpu().flatten()))
        chan1 = np.concatenate((chan0, images[:, 1, :, :].cpu().flatten()))
        chan2 = np.concatenate((chan0, images[:, 2, :, :].cpu().flatten()))
        
    means = [np.mean(chan0), np.mean(chan1), np.mean(chan2)]
    stds  = [np.std(chan0), np.std(chan1), np.std(chan2)]
    
    return means, stds