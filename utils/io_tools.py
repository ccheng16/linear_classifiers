"""Input and output helpers to load in data.
"""

import numpy as np
import skimage
import os
from skimage import io


def read_dataset(data_txt_file, image_data_path):
    """Read data into a Python dictionary.
    Args:
        data_txt_file(str): path to the data txt file.
        image_data_path(str): path to the image directory.
    Returns:
        data(dict): A Python dictionary with keys 'image' and 'label'.
            The value of dict['image'] is a numpy array of dimension (N,28,28)
            containing the loaded images.

            The value of dict['label'] is a numpy array of dimension (N,)
            containing the loaded label.

            N is the number of examples in the data split, the examples should
            be stored in the same order as in the txt file.
    """

    f = open(data_txt_file, 'r')
    images = []
    labels = []
    for line in f.readlines():
        filename = line.split()[0]
        label = line.split()[1]
        path_to_image = os.path.join(image_data_path, filename)
        image = io.imread(path_to_image)
        images.append(image)
        labels.append(int(label))
    f.close()
    data = {}
    data['image'] = np.array(images)
    print('shape of images is {}'.format(data['image'].shape))
    data['label'] = np.array(labels)
    print('shape of labels is {}'.format(data['label'].shape))
    return data


def write_dataset(data_txt_file, data):
    """Write python dictionary data into csv format for kaggle.
    Args:
        data_txt_file(str): path to the data txt file.
        data(dict): A Python dictionary with keys 'image' and 'label',
          (see descriptions above).

    To submit to Kaggle, the submission format is in csv. The uploaded file should be of 8001 lines, format illustrated below:

        Id,Prediction
        test_00000.png,1
        ... test_01234.png,-1
        ... test_07999.png,-1

    """
    images = data['image']
    labels = data['label']
    data_file_path = os.path.abspath(data_txt_file)
    print('file path is {}'.format(data_file_path))
    with open(data_file_path, 'w') as f:
        f.write('Id,Prediction\n')
        i = 0
        for p in data['prediction']:
            f.write('test_%05d.png,%d\n'%(i, p))
            i += 1