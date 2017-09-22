"""
Train model and eval model helpers.
"""
from __future__ import print_function

import numpy as np
from models.linear_regression import LinearRegression


def train_model(data, model, learning_rate=0.001, batch_size=16,
                num_steps=1000, shuffle=True):
    """Implements the training loop of stochastic gradient descent.

    Performs stochastic gradient descent with the indicated batch_size.
    If shuffle is true:
        Shuffle data at every epoch, including the 0th epoch.
    If the number of example is not divisible by batch_size, the last batch
    will simply be the remaining examples.

    Args:
        data(dict): Data loaded from io_tools
        model(LinearModel): Initialized linear model.
        learning_rate(float): Learning rate of your choice
        batch_size(int): Batch size of your choice.
        num_steps(int): Number of steps to run the updated.
        shuffle(bool): Whether to shuffle data at every epoch.
    Returns:
        model(LinearModel): Returns a trained model.
    """
    # Perform gradient descent.
    N = data['image'].shape[0]
    batch_epoch_num = N // batch_size + 1 * (N % batch_size != 0)
    global_step = 0
    for i in range(num_steps):
        global_step += 1
        i %= batch_epoch_num
        if shuffle:
            p = np.random.permutation(N)
            indices = p[:batch_size]
        else:
            indices = np.arange(i * batch_size, (i + 1) * batch_size)
        # print('Step {}'.format(global_step))
        update_step(data['image'][indices], data['label'][indices], model=model, learning_rate=learning_rate)
    return model


def update_step(image_batch, label_batch, model, learning_rate):
    """Performs on single update step, (i.e. forward then backward).
    Args:
        image_batch(numpy.ndarray): input data of dimension (N, ndims).
        label_batch(numpy.ndarray): label data of dimension (N,).
        model(LinearModel): Initialized linear model.
    """
    f = model.forward(image_batch)
    model.w -= learning_rate * model.backward(f, label_batch)

def eval_model(data, model):
    """Performs evaluation on a dataset.
    Args:
        data(dict): Data loaded from io_tools.
        model(LinearModel): Initialized linear model.
    Returns:
        loss(float): model loss on data.
        acc(float): model accuracy on data.
    """
    N = data['image'].shape[0]
    f = model.forward(data['image'])
    labels = data['label']
    y_predict = model.predict(f)
    loss = model.loss(f, labels)
    acc = np.mean(labels == y_predict)
    # data['label'] = y_predict
    return loss, acc, y_predict