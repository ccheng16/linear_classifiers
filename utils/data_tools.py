"""
Implements feature extraction and other data processing helpers.
"""

import numpy as np
import skimage
from skimage import filters
from numpy import fft


def preprocess_data(data, process_method='default'):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
        process_method(str): processing methods needs to support
          ['raw', 'default'].
        if process_method is 'raw'
          1. Convert the images to range of [0, 1]
          2. Remove mean.
          3. Flatten images, data['image'] is converted to dimension (N, 28*28)
        if process_method is 'default':
          1. Convert images to range [0,1]
          2. Apply laplacian filter with window size of 11x11. (use skimage)
          3. Remove mean.
          4. Flatten images, data['image'] is converted to dimension (N, 28*28)

    Returns:
        data(dict): Apply the described processing based on the process_method
        str to data['image'], then return data.
    """
    if process_method == 'default':
        data['image'] = (data['image'] / 255).astype(float)
        images = data['image']
        for img in images:
            img = filters.laplace(img, 11)
        data['images'] = images
        data = remove_data_mean(data)
        images_shape = data['image'].shape
        data['image'] = np.reshape(data['image'], (images_shape[0], images_shape[1] * images_shape[2]))
        # img_max = np.amax(data['image'], axis=1)
        # data['image'] /= img_max.reshape((img_max.shape[0], 1))

    elif process_method == 'raw':
        data['image'] = (data['image'] / 255).astype(float)
        data = remove_data_mean(data)
        images_shape = data['image'].shape
        data['image'] = np.reshape(data['image'], (images_shape[0], images_shape[1] * images_shape[2]))
        # img_max = np.amax(data['image'], axis=1)

    elif process_method == 'custom':
        data['image'] = (data['image'] / 255).astype(float)
        data = remove_data_mean(data)
        for img in data['image']:
            gx, gy = np.gradient(img)
            img += (gx ** 2 + gy ** 2)
        images_shape = data['image'].shape
        data['image'] = np.reshape(data['image'], (images_shape[0], images_shape[1] * images_shape[2]))
        # img_max = np.amax(data['image'], axis=1)

    return data


def compute_image_mean(data):
    """ Computes mean image.
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        image_mean(numpy.ndarray): Average across the example dimension.
    """
    images = data['image']
    image_mean = np.mean(images, axis=0)
    return image_mean


def remove_data_mean(data):
    """
    Args:
        data(dict): Python dict loaded using io_tools.
    Returns:
        data(dict): Remove mean from data['image'] and return data.
    """
    data['image'] -= compute_image_mean(data)
    return data
