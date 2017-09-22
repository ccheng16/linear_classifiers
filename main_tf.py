"""Main function for train, eval, and test.
"""

from __future__ import print_function
from __future__ import absolute_import

import numpy as np
import tensorflow as tf

from models.linear_regression_tf import LinearRegressionTf
from models.logistic_regression_tf import LogisticRegressionTf
from models.support_vector_machine_tf import SupportVectorMachineTf

from train_eval_model_tf import train_model, eval_model
from utils.io_tools import read_dataset, write_dataset
from utils.data_tools import preprocess_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_string('feature_type', 'default', 'Feature type, supports ['']')
flags.DEFINE_string('model_type', 'linear', 'Feature type, supports ['']')


def main(_):
    """High level pipeline.

    This script performs the trainsing, evaling and testing state of the model.
    """
    learning_rate = FLAGS.learning_rate
    feature_type = FLAGS.feature_type
    model_type = FLAGS.model_type

    # Load dataset.
    data = read_dataset('data/assignment1_data/train_lab.txt', 'data/assignment1_data/image_data')

    # Data Processing.
    data = preprocess_data(data, feature_type)

    # Initialize model.
    ndim = data['image'].shape[1]
    if model_type == 'linear':
        model = LinearRegressionTf(ndim, 'ones')
    elif model_type == 'logistic':
        model = LogisticRegressionTf(ndim, 'uniform')
    elif model_type == 'svm':
        model = SupportVectorMachineTf(ndim, 'gaussian')

    # Train Model.
    model = train_model(data, model, learning_rate, num_steps=40000)

    # Eval Model.
    data_val = read_dataset('data/assignment1_data/val_lab.txt', 'data/assignment1_data/image_data')
    data_val = preprocess_data(data_val, feature_type)
    loss, acc,_ = eval_model(data_val, model)
    print('loss is {}'.format(loss))
    print('acc is {}'.format(acc))

    # Test Model.
    data_test = read_dataset('data/assignment1_data/test_lab.txt', 'data/assignment1_data/image_data')
    data_test = preprocess_data(data_test, feature_type)
    _, _, y_predict = eval_model(data_test, model)
    # Generate Kaggle output.
    data_test['prediction'] = y_predict
    write_dataset('data/assignment1_data/predictions.csv', data_test)

if __name__ == '__main__':
    tf.app.run()
