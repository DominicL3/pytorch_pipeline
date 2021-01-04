import numpy as np

"""
Collection of preprocessing functions for image classification or segmentation.
"""

def normalize(x):
    """
    Normalizes input array by dividing out the max value, bringing the range to 0-1.

    Args:
        x (array): Array to be normalized.

    Returns:
        x (array): Array after normalization, now in 0-1 range.
    """
    x = x / np.max(x)
    return x