import os
from utils import exampleutils

import torch
from torch import Tensor
import numpy as np


def calculate_metric_measure(metrics_tensor: Tensor, measure: str) -> dict:
    """ Calculates measure values (e.g., average, variance, etc.) of the given metric (e.g., loss) for each example and returns a dictionary with examples as keys and the average metric for each example as their values.
    """


    variances_means = torch.var_mean(metrics_tensor, dim=1)
    examples_importances = np.zeros_like(variances_means)
    if measure.lower() == 'average':
        examples_importances = variances_means[1].detach().cpu().numpy()
    elif measure.lower() == 'variance':
        # if there is only one value per example, some measures, e.g., variance, will be NaN
        # return the average instead
        if metrics_tensor.shape[1] == 1:
            examples_importances = variances_means[1].detach().cpu().numpy()
        else:
            examples_importances = variances_means[0].detach().cpu().numpy()
    
    # TODO add test to ensure examples_means.argmax() is the first item of the sorted dict
    # print(examples_means.argmax())
    examples_dict = {index : examples_importances[index] for index in range(0, len(examples_importances))}
    return examples_dict
    